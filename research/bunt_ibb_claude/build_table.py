"""
Build everything the pre-registered bunt / walk-candidate study needs before
any API call (PREREG.md in this directory).

  python research/bunt_ibb_claude/build_table.py --states <dir with pa_*.parquet> --out <work dir>

Writes to --out:
  rows_bunt.parquet, rows_obp.parquet  one row per scored unit with split,
                      label y, model A's probability (p_a) and, for control
                      rows, A refitted without April-June 2026 (p_a_ctrl)
  payload_B.jsonl     Claude input without the model's estimate
  payload_C.jsonl     the same plus the model's estimate
  sample_payload.json one example of each part and arm
  names.json          StatsAPI names of players and teams (cached)

Prints counts only (no outcome of the test or control rows).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 20261026
TRAIN_START, TEST_START = "2023-03-30", "2026-07-01"
CTRL_START, CTRL_END = "2026-04-01", "2026-06-30"
PILOT_START = "2026-06-01"
N_CTRL, N_PILOT = 300, 20
BASE = "https://statsapi.mlb.com/api/v1"

ON_BASE = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch"}
MIN_LEAGUE = {"bunt": 100, "obp": 2000}       # earlier-date units needed before the season's own rate is used
SHRINK = {"bunt_batter": 5.0, "obp_batter": 200.0, "obp_pitcher": 200.0}


# ---------------------------------------------------------------- tables

def load_pa(states: Path) -> pd.DataFrame:
    pa = pd.concat([pd.read_parquet(p) for p in sorted(states.glob("pa_*.parquet"))], ignore_index=True)
    pa["date"] = pd.to_datetime(pa["date"])
    bh = pa.half == "bottom"
    pa["bat_diff"] = np.where(bh, pa.home_score - pa.away_score, pa.away_score - pa.home_score)
    pa["off_team"] = np.where(bh, pa.home_team, pa.away_team)
    pa["def_team"] = np.where(bh, pa.away_team, pa.home_team)
    return pa.sort_values(["date", "game_pk", "at_bat_index"]).reset_index(drop=True)


def bunt_units(pa: pd.DataFrame) -> pd.DataFrame:
    """Completed plate appearances that ended on a bunt (description contains
    "bunt"), 0 or 1 out, a runner on first and/or second, third base empty.
    y = 1 when the lead runner (on second if there is one, else on first)
    ends the play on a later base or scores; else 0."""
    b = pa[pa.is_bunt & pa.pa_complete & (pa.outs < 2) & pa.on_3b.isna()
           & (pa.on_1b.notna() | pa.on_2b.notna())].copy()
    # Bases are read by occupancy, not by runner id: a pinch runner put in
    # after the play shows up under a new id in the post-play state. The
    # batter is never counted as the advanced runner. Third base is empty
    # before the play, so a run on the play means the lead runner scored.
    from_second = b.on_2b.notna()
    occ = lambda col: b[col].notna() & (b[col] != b.batter)
    scored = b.runs_on_play > 0
    b["y"] = np.where(from_second, occ("post_3b") | scored,
                      occ("post_2b") | occ("post_3b") | scored).astype(int)
    b["lead_from_second"] = from_second.astype(int)
    b["two_runners"] = (b.on_1b.notna() & b.on_2b.notna()).astype(int)
    return b


def obp_units(pa: pd.DataFrame) -> pd.DataFrame:
    """Completed plate appearances, 7th inning or later, first base empty, a
    runner on second and/or third, not an intentional walk or catcher's
    interference. y = 1 for single, double, triple, home run, walk, hit by pitch."""
    o = pa[pa.pa_complete & (pa.inning >= 7) & pa.on_1b.isna() & (pa.on_2b.notna() | pa.on_3b.notna())
           & ~pa.event_type.isin(["intent_walk", "catcher_interf"])].copy()
    o["y"] = o.event_type.isin(ON_BASE).astype(int)
    o["runner_2b"] = o.on_2b.notna().astype(int)
    o["runner_3b"] = o.on_3b.notna().astype(int)
    return o


def pa_obp_label(pa: pd.DataFrame) -> pd.DataFrame:
    """Every completed PA (except IBB / catcher's interference) as a history
    source for batter and pitcher on-base records."""
    h = pa[pa.pa_complete & ~pa.event_type.isin(["intent_walk", "catcher_interf"])].copy()
    h["ob"] = h.event_type.isin(ON_BASE).astype(int)
    return h


# ---------------------------------------------------------------- causal records

def league_rate(hist: pd.DataFrame, y: str, min_n: int) -> pd.DataFrame:
    """Per date: this season's rate on earlier dates (n, successes, rate used).
    The previous season's full rate is used until the season has min_n units."""
    hist = hist.assign(season=hist.date.dt.year)
    d = hist.groupby(["season", "date"])[y].agg(["size", "sum"]).reset_index().sort_values("date")
    d["n_before"] = d.groupby("season")["size"].cumsum() - d["size"]
    d["s_before"] = d.groupby("season")["sum"].cumsum() - d["sum"]
    full = hist.groupby("season")[y].mean()
    prev = d.season.map(lambda s: full.get(s - 1, np.nan))
    d["rate"] = np.where(d.n_before >= min_n, d.s_before / d.n_before.clip(lower=1), prev)
    return d[["date", "n_before", "s_before", "rate"]].rename(
        columns={"n_before": "lg_n_before", "s_before": "lg_s_before", "rate": "lg_rate"})


def _per_date_counts(hist: pd.DataFrame, who: str, y: str, dates: pd.DataFrame) -> pd.DataFrame:
    """Counts strictly before each (who, date) in `dates`, including dates on
    which `who` has no history row (merge_asof on the history's own dates
    would then carry that day's own counts forward: it is avoided by
    computing cumulative counts through the previous history date)."""
    daily = (hist.groupby([who, "date"])[y].agg(["size", "sum"]).rename(
        columns={"size": "n", "sum": "s"}).reset_index().sort_values([who, "date"]))
    daily["n_thru"] = daily.groupby(who).n.cumsum()
    daily["s_thru"] = daily.groupby(who).s.cumsum()
    q = dates[[who, "date"]].drop_duplicates().copy()
    q["_q"] = q["date"] - pd.Timedelta(days=1)          # through the day before
    q = q.sort_values("_q")
    daily = daily.rename(columns={"date": "_q"}).sort_values("_q")
    m = pd.merge_asof(q.assign(**{who: q[who].astype("int64")}),
                      daily[[who, "_q", "n_thru", "s_thru"]].assign(**{who: daily[who].astype("int64")}),
                      on="_q", by=who, direction="backward")
    m["n_before"] = m.n_thru.fillna(0).astype(int)
    m["s_before"] = m.s_thru.fillna(0).astype(int)
    return m[[who, "date", "n_before", "s_before"]]


def add_records(units: pd.DataFrame, hist: pd.DataFrame, who: str, y: str, prefix: str) -> pd.DataFrame:
    c = _per_date_counts(hist, who, y, units.assign(**{who: units[who].astype("int64")}))
    out = units.assign(**{who: units[who].astype("int64")}).merge(c, on=[who, "date"], how="left")
    out[f"{prefix}_n_before"] = out.pop("n_before").astype(int)
    out[f"{prefix}_s_before"] = out.pop("s_before").astype(int)
    return out


# ---------------------------------------------------------------- features / model A

def _logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def features(part: str, u: pd.DataFrame) -> pd.DataFrame:
    r = u.lg_rate.to_numpy()
    X = pd.DataFrame(index=u.index)
    if part == "bunt":
        k = SHRINK["bunt_batter"]
        X["batter_rate_logit_minus_lg"] = _logit((u.bat_s_before + k * r) / (u.bat_n_before + k)) - _logit(r)
        X["batter_log1p_bunts"] = np.log1p(u.bat_n_before)
        X["batter_log1p_pa"] = np.log1p(u.batpa_n_before)
        X["batter_obp_logit_minus_lg"] = _logit((u.batpa_s_before + 200 * 0.31) / (u.batpa_n_before + 200)) - _logit(0.31)
        X["lead_from_second"] = u.lead_from_second
        X["two_runners"] = u.two_runners
        X["one_out"] = (u.outs == 1).astype(int)
        X["late_close"] = ((u.inning >= 7) & (u.bat_diff.abs() <= 1)).astype(int)
    else:
        kb, kp = SHRINK["obp_batter"], SHRINK["obp_pitcher"]
        X["batter_rate_logit_minus_lg"] = _logit((u.bat_s_before + kb * r) / (u.bat_n_before + kb)) - _logit(r)
        X["pitcher_rate_logit_minus_lg"] = _logit((u.pit_s_before + kp * r) / (u.pit_n_before + kp)) - _logit(r)
        X["batter_log1p_pa"] = np.log1p(u.bat_n_before)
        X["pitcher_log1p_bf"] = np.log1p(u.pit_n_before)
        X["runner_2b"] = u.runner_2b
        X["runner_3b"] = u.runner_3b
        X["outs1"] = (u.outs == 1).astype(int)
        X["outs2"] = (u.outs == 2).astype(int)
        X["extra_innings"] = (u.inning > u.scheduled_innings).astype(int)
        X["bat_diff_clip"] = u.bat_diff.clip(-4, 4)
        X["close"] = (u.bat_diff.abs() <= 1).astype(int)
    return X.astype(float)


def fit_a(part: str, rows: pd.DataFrame, C: float = 1.0):
    """L2 logistic regression with logit(league rate) as an offset
    (coefficient fixed at 1); standardized features; intercept unpenalized."""
    X = features(part, rows)
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = np.column_stack([np.ones(len(X)), ((X - mu) / sd).to_numpy()])
    y = rows.y.to_numpy().astype(float)
    off = _logit(rows.lg_rate)

    def f(w):
        eta = off + Z @ w
        p = 1 / (1 + np.exp(-eta))
        loss = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 / C * np.sum(w[1:] ** 2)
        g = Z.T @ (p - y)
        g[1:] += w[1:] / C
        return loss, g

    res = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        sys.exit(f"model A ({part}) did not converge: {res.message}")
    return res.x, mu, sd


def predict(part: str, model, rows: pd.DataFrame) -> np.ndarray:
    w, mu, sd = model
    X = features(part, rows)
    Z = np.column_stack([np.ones(len(X)), ((X - mu) / sd).to_numpy()])
    return 1 / (1 + np.exp(-(_logit(rows.lg_rate) + Z @ w)))


# ---------------------------------------------------------------- names / payloads

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-win-probability/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def names_for(ids: set[int], teams: set[int], cache: Path) -> dict:
    data = json.loads(cache.read_text()) if cache.exists() else {"people": {}, "teams": {}}
    need = sorted(i for i in ids if str(i) not in data["people"])
    for k in range(0, len(need), 100):
        chunk = ",".join(str(i) for i in need[k:k + 100])
        for p in fetch(f"{BASE}/people?personIds={chunk}").get("people", []):
            data["people"][str(p["id"])] = p["fullName"]
    if any(str(t) not in data["teams"] for t in teams):
        for t in fetch(f"{BASE}/teams?sportId=1&season=2026").get("teams", []):
            data["teams"][str(t["id"])] = t["name"]
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    return data


POPULATION = {
    "bunt": ("Completed plate appearances in MLB regular-season games that ended on a bunt "
             "(sacrifice, bunt hit, bunt out, or strikeout on a foul bunt) with 0 or 1 out, a "
             "runner on first and/or second and third base empty. Success = the lead runner "
             "(the runner on second if there is one, otherwise the runner on first) ends the "
             "play on a later base or scores. Failure = the lead runner is put out or stays."),
    "obp": ("Completed plate appearances in MLB regular-season games in the 7th inning or "
            "later with first base empty and a runner on second and/or third. Intentional "
            "walks and catcher's interference are excluded. Success = the batter reaches "
            "base by a hit, walk or hit by pitch; anything else (including reaching on an "
            "error or a fielder's choice) = failure."),
}
LEAGUE_NOTE = {
    "bunt": ("rate of earlier dates this season in this population; the previous season's "
             "rate until the season has 100 such bunts"),
    "obp": ("on-base rate (hit, walk, hit by pitch) of all completed plate appearances on "
            "earlier dates this season, excluding intentional walks and catcher's "
            "interference; the previous season's rate until the season has 2000 "
            "plate appearances"),
}
QUESTION = {
    "bunt": "Probability that the lead runner advances (success as defined).",
    "obp": "Probability that the batter reaches base (success as defined).",
}


def payload(part: str, r, nm: dict, with_model: bool) -> dict:
    who = lambda i: nm["people"].get(str(int(i)))
    p = {
        "part": part,
        "date": r.date.strftime("%Y-%m-%d"),
        "population": POPULATION[part],
        "question": QUESTION[part],
        "league_this_season_before_today": {
            "units": int(r.lg_n_before), "successes": int(r.lg_s_before),
            "success_rate_used": round(float(r.lg_rate), 3),
            "note": LEAGUE_NOTE[part],
        },
        "situation": {
            "inning": int(r.inning), "half": r.half, "outs": int(r.outs),
            "runner_on_first": bool(pd.notna(r.on_1b)), "runner_on_second": bool(pd.notna(r.on_2b)),
            "runner_on_third": bool(pd.notna(r.on_3b)),
            "batting_team_score_minus_fielding_team_score": int(r.bat_diff),
        },
        "teams": {"offense": nm["teams"].get(str(int(r.off_team))), "defense": nm["teams"].get(str(int(r.def_team)))},
        "batter": {"name": who(r.batter)},
        "pitcher": {"name": who(r.pitcher)},
    }
    if part == "bunt":
        p["batter"].update({
            "bunts_in_this_population_since_2021_before_today": int(r.bat_n_before),
            "successes_in_those_bunts": int(r.bat_s_before),
            "plate_appearances_since_2021_before_today": int(r.batpa_n_before),
            "times_on_base_in_those_plate_appearances": int(r.batpa_s_before)})
    else:
        p["batter"].update({
            "plate_appearances_since_2021_before_today": int(r.bat_n_before),
            "times_on_base_in_those_plate_appearances": int(r.bat_s_before)})
        p["pitcher"].update({
            "batters_faced_since_2021_before_today": int(r.pit_n_before),
            "times_on_base_allowed": int(r.pit_s_before)})
    if with_model:
        p["statistical_model_estimate"] = {
            "p_success": round(float(r.p_model), 3),
            "what_it_is": ("L2 logistic regression on the situation, the league rate and the "
                           "shrunk records above; trained on this population from "
                           f"{TRAIN_START} to the day before "
                           f"{CTRL_START if r.split == 'control' else TEST_START}."),
        }
    return p


# ---------------------------------------------------------------- main

def build_part(part: str, pa: pd.DataFrame, hist: pd.DataFrame, rng) -> pd.DataFrame:
    if part == "bunt":
        u = bunt_units(pa)
        u = add_records(u, u, "batter", "y", "bat")
        u = add_records(u, hist, "batter", "ob", "batpa")
        lg = league_rate(u[["date", "y"]], "y", MIN_LEAGUE["bunt"])
    else:
        u = obp_units(pa)
        u = add_records(u, hist, "batter", "ob", "bat")
        u = add_records(u, hist, "pitcher", "ob", "pit")
        lg = league_rate(hist[["date", "ob"]], "ob", MIN_LEAGUE["obp"])
    u = u.merge(lg, on="date", how="left")
    u = u.sort_values(["date", "game_pk", "at_bat_index"]).reset_index(drop=True)
    u["custom_id"] = [f"{part}-{r.game_pk}-{r.at_bat_index}" for r in u.itertuples()]
    assert u.custom_id.is_unique

    u["split"] = "none"
    u.loc[(u.date >= TRAIN_START) & (u.date < TEST_START), "split"] = "train"
    u.loc[u.date >= TEST_START, "split"] = "test"
    ctrl_pool = u.index[(u.date >= CTRL_START) & (u.date <= CTRL_END)]
    ctrl = rng.choice(ctrl_pool, size=N_CTRL, replace=False)
    pilot_pool = u.index[(u.date >= PILOT_START) & (u.date <= CTRL_END)].difference(ctrl)
    pilot = rng.choice(pilot_pool, size=N_PILOT, replace=False)

    train = u[u.split == "train"]
    if u[u.split != "none"].lg_rate.isna().any():
        sys.exit(f"{part}: league rate missing")
    model_a = fit_a(part, train)
    model_ctrl = fit_a(part, train[train.date < CTRL_START])
    u.loc[ctrl, "split"] = "control"       # control and pilot rows stay in A's training
    u.loc[pilot, "split"] = "pilot"
    u["p_a"] = predict(part, model_a, u)
    u["p_a_ctrl"] = np.where(u.split == "control", predict(part, model_ctrl, u), np.nan)
    u["p_model"] = np.where(u.split == "control", u.p_a_ctrl, u.p_a)
    u["part"] = part
    return u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pa = load_pa(a.states)
    hist = pa_obp_label(pa)
    rng = np.random.default_rng(SEED)
    parts = {part: build_part(part, pa, hist, rng) for part in ("bunt", "obp")}

    use = pd.concat([u[u.split.isin(["test", "control", "pilot"])] for u in parts.values()])
    nm = names_for(set(use.batter.astype(int)) | set(use.pitcher.astype(int)),
                   set(use.off_team.astype(int)) | set(use.def_team.astype(int)), a.out / "names.json")
    missing = sum(1 for r in use.itertuples() for k in ("batter", "pitcher")
                  if nm["people"].get(str(int(getattr(r, k)))) is None)
    missing += sum(1 for r in use.itertuples() for k in ("off_team", "def_team")
                   if nm["teams"].get(str(int(getattr(r, k)))) is None)
    if missing:
        sys.exit(f"{missing} names missing - not writing payloads")

    for arm, with_model in (("B", False), ("C", True)):
        with open(a.out / f"payload_{arm}.jsonl", "w", encoding="utf-8") as f:
            for part, u in parts.items():
                for r in u[u.split.isin(["test", "control", "pilot"])].itertuples():
                    f.write(json.dumps({"custom_id": r.custom_id, "part": part, "split": r.split,
                                        "payload": payload(part, r, nm, with_model)},
                                       ensure_ascii=False, sort_keys=True) + "\n")
    ex = {}
    for part, u in parts.items():
        r = next(u[u.split == "pilot"].itertuples())
        ex[part] = {"B": payload(part, r, nm, False), "C": payload(part, r, nm, True)}
    (a.out / "sample_payload.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
    keep = ["custom_id", "part", "split", "game_pk", "date", "batter", "pitcher", "y",
            "p_a", "p_a_ctrl", "p_model", "lg_rate", "bat_n_before"]
    for part, u in parts.items():
        u[keep].to_parquet(a.out / f"rows_{part}.parquet", index=False)
        print(part, u.split.value_counts().to_dict(),
              "test games", int(u[u.split == "test"].game_pk.nunique()))


if __name__ == "__main__":
    main()
