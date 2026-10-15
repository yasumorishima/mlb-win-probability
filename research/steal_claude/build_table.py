"""
Build everything the pre-registered steal study needs before any API call.

  python research/steal_claude/build_table.py --states data/states --hf data/hf --out research/steal_claude/work

Writes to --out:
  rows.parquet        one row per attempt with split (train / control / pilot / test),
                      model A's probability (p_a) and, for control rows, the
                      probability of A refitted without April-June 2026 (p_a_ctrl)
  payload_B.jsonl     Claude input without the model's estimate (control, pilot, test)
  payload_C.jsonl     the same plus the model's estimate
  sample_payload.json one example of each, for the frozen commit
  names.json          StatsAPI names of players and teams (cached)

Splits (PREREG.md):
  train    2023-03-30 .. 2026-06-30
  test     2026-07-01 .. end of the 2026 regular season
  control  300 random attempts 2026-04-01 .. 2026-06-30 (seed 20261015)
  pilot    40 random June 2026 attempts not in control (seed 20261015)
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from steal_features import add_features, design, load_attempts  # noqa: E402

SEED = 20261015
TRAIN_START, TEST_START = "2023-03-30", "2026-07-01"
CTRL_START, CTRL_END = "2026-04-01", "2026-06-30"
BASE = "https://statsapi.mlb.com/api/v1"


def _offset(rows):
    """logit of this season's league rate before the date: the level the
    model starts from (coefficient fixed at 1), so the model is calibrated
    to the current season and learns only departures from it."""
    r = rows.season_rate_before.clip(0.01, 0.99).to_numpy()
    return np.log(r / (1 - r))


def fit_a(rows: pd.DataFrame, C: float = 1.0):
    """L2 logistic regression with an offset (sklearn has no offsets):
    minimize sum log loss + ||beta||^2 / (2C), intercept unpenalized."""
    X, fill = design(rows)
    X = X.drop(columns=["season_rate_logit"])
    sc = StandardScaler().fit(X)
    Z = np.column_stack([np.ones(len(X)), sc.transform(X)])
    y = rows.success.to_numpy().astype(float)
    off = _offset(rows)

    def f(w):
        eta = off + Z @ w
        p = 1 / (1 + np.exp(-eta))
        loss = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 / C * np.sum(w[1:] ** 2)
        grad = Z.T @ (p - y)
        grad[1:] += w[1:] / C
        return loss, grad

    res = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        sys.exit(f"model A did not converge: {res.message}")
    return res.x, sc, fill


def predict(model, rows):
    w, sc, fill = model
    X, _ = design(rows, fill)
    X = X.drop(columns=["season_rate_logit"])
    Z = np.column_stack([np.ones(len(X)), sc.transform(X)])
    return 1 / (1 + np.exp(-(_offset(rows) + Z @ w)))


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-win-probability/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def names_for(ids: set[int], teams: set[int], cache: Path) -> dict:
    data = json.loads(cache.read_text()) if cache.exists() else {"people": {}, "teams": {}}
    need = [i for i in ids if str(i) not in data["people"]]
    for k in range(0, len(need), 100):
        chunk = ",".join(str(i) for i in need[k:k + 100])
        for p in fetch(f"{BASE}/people?personIds={chunk}").get("people", []):
            data["people"][str(p["id"])] = p["fullName"]
    if any(str(t) not in data["teams"] for t in teams):
        for t in fetch(f"{BASE}/teams?sportId=1&season=2026").get("teams", []):
            data["teams"][str(t["id"])] = t["name"]
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    return data


def reference(hf: Path) -> dict:
    """League mean / sd of the previous-season (2025) Savant measures."""
    sp = pd.read_parquet(hf / "sprint_speed.parquet")
    sp = sp[sp.season == 2025].sprint_speed
    ca = pd.read_parquet(hf / "catcher.parquet")
    ca = ca[ca.season == 2025]
    r = lambda s: {"mean": round(float(s.mean()), 2), "sd": round(float(s.std()), 2)}
    return {"sprint_speed_ft_per_s": r(sp), "pop_time_2b_s": r(ca.pop_2b_sba.dropna()),
            "arm_strength_mph": r(ca.maxeff_arm_2b_3b_sba.dropna())}


def _num(x, nd=2):
    return None if pd.isna(x) else round(float(x), nd)


def payload(r, nm: dict, ref: dict, with_model: bool) -> dict:
    off_team = r.home_team if r.half == "bottom" else r.away_team
    def_team = r.away_team if r.half == "bottom" else r.home_team
    p = {
        "date": r.date.strftime("%Y-%m-%d"),
        "population": ("Attempts to steal second or third base that happened on a pitch, "
                       "in MLB regular-season games. Steals of home, pickoffs and "
                       "attempts not on a pitch are excluded. Success = stolen base "
                       "credited; caught stealing (including on an error) = failure."),
        "league_this_season_before_today": {
            "attempts": int(r.season_att_before), "successes": int(r.season_succ_before),
            "success_rate_used": round(float(r.season_rate_before), 3),
            "note": ("rate of earlier dates this season; previous season's rate "
                     "until 300 attempts") },
        "situation": {
            "target_base": "third" if r.to_third else "second",
            "outs": int(r.outs_before), "balls": int(r.balls), "strikes": int(r.strikes),
            "pickoff_attempts_and_stepoffs_this_PA_before_this_pitch": int(r.disengagements_before),
            "inning": int(r.inning), "half": r.half,
            "batting_team_score_minus_fielding_team_score": int(r.bat_diff),
            "pitcher_throws": r.pitch_hand,
        },
        "teams": {"offense": nm["teams"].get(str(off_team)), "defense": nm["teams"].get(str(def_team))},
        "runner": {
            "name": nm["people"].get(str(int(r.runner))),
            "sprint_speed_previous_season_ft_per_s": _num(r.runner_sprint_prev, 1),
            "attempts_since_2021_before_today": int(r.runner_att_before),
            "successes_since_2021_before_today": int(r.runner_succ_before),
        },
        "pitcher": {
            "name": nm["people"].get(str(int(r.pitcher))),
            "attempts_against_since_2021_before_today": int(r.pitcher_att_before),
            "successes_against_since_2021_before_today": int(r.pitcher_succ_before),
        },
        "catcher": {
            "name": nm["people"].get(str(int(r.catcher))),
            "pop_time_to_second_previous_season_s": _num(r.catcher_pop_prev, 2),
            "arm_strength_previous_season_mph": _num(r.catcher_arm_prev, 1),
            "attempts_against_since_2021_before_today": int(r.catcher_att_before),
            "successes_against_since_2021_before_today": int(r.catcher_succ_before),
        },
        "league_reference_previous_season": ref,
    }
    if with_model:
        p["statistical_model_estimate"] = {
            "p_success": round(float(r.p_model), 3),
            "what_it_is": ("L2 logistic regression on the situation, the league rate, "
                           "the previous-season Savant measures and the shrunk "
                           "attempt records above; trained on attempts from "
                           f"{TRAIN_START} to the day before {TEST_START if r.split != 'control' else CTRL_START}."),
        }
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    att = add_features(load_attempts(a.states), a.hf)
    pa = pd.concat([pd.read_parquet(f, columns=["game_pk", "home_team", "away_team"])
                    for f in sorted(a.states.glob("pa_*.parquet"))]).drop_duplicates("game_pk")
    att = att.merge(pa, on="game_pk", how="left")

    att["split"] = "none"
    att.loc[(att.date >= TRAIN_START) & (att.date < TEST_START), "split"] = "train"
    att.loc[att.date >= TEST_START, "split"] = "test"
    rng = np.random.default_rng(SEED)
    ctrl_pool = att.index[(att.date >= CTRL_START) & (att.date <= CTRL_END)]
    ctrl = rng.choice(ctrl_pool, size=300, replace=False)
    pilot_pool = att.index[(att.date >= "2026-06-01") & (att.date <= CTRL_END)].difference(ctrl)
    pilot = rng.choice(pilot_pool, size=40, replace=False)

    train = att[att.split == "train"]
    model_a = fit_a(train)
    model_ctrl = fit_a(train[train.date < CTRL_START])

    att.loc[ctrl, "split"] = "control"       # control and pilot rows stay in A's training
    att.loc[pilot, "split"] = "pilot"
    att["p_a"] = predict(model_a, att)
    att["p_a_ctrl"] = np.where(att.split == "control", predict(model_ctrl, att), np.nan)
    # the estimate shown to Claude in arm C: the model that did not see the row
    att["p_model"] = np.where(att.split == "control", att.p_a_ctrl, att.p_a)

    use = att[att.split.isin(["test", "control", "pilot"])].copy()
    nm = names_for(set(use.runner.astype(int)) | set(use.pitcher.astype(int)) | set(use.catcher.astype(int)),
                   set(use.home_team) | set(use.away_team), a.out / "names.json")
    ref = reference(a.hf)
    missing = sum(1 for r in use.itertuples() for k in ("runner", "pitcher", "catcher")
                  if nm["people"].get(str(int(getattr(r, k)))) is None)
    if missing:
        sys.exit(f"{missing} player names missing - not writing payloads")

    use["custom_id"] = [f"{r.game_pk}-{r.at_bat_index}-{r.event_index}-{int(r.runner)}-{r.event_type}" for r in use.itertuples()]
    assert use.custom_id.is_unique
    for arm, with_model in (("B", False), ("C", True)):
        with open(a.out / f"payload_{arm}.jsonl", "w", encoding="utf-8") as f:
            for r in use.itertuples():
                f.write(json.dumps({"custom_id": r.custom_id, "split": r.split,
                                    "payload": payload(r, nm, ref, with_model)},
                                   ensure_ascii=False, sort_keys=True) + "\n")
    ex = use[use.split == "pilot"].iloc[0]
    (a.out / "sample_payload.json").write_text(json.dumps(
        {"B": payload(ex, nm, ref, False), "C": payload(ex, nm, ref, True)}, ensure_ascii=False, indent=1))
    keep = ["custom_id", "split", "game_pk", "date", "runner", "pitcher", "catcher", "success",
            "p_a", "p_a_ctrl", "p_model", "season_rate_before", "runner_sprint_prev", "catcher_pop_prev",
            "runner_att_before"]
    att_out = att.copy()
    att_out["custom_id"] = [f"{r.game_pk}-{r.at_bat_index}-{r.event_index}-{int(r.runner)}-{r.event_type}" for r in att_out.itertuples()]
    att_out[keep].to_parquet(a.out / "rows.parquet", index=False)
    print(att.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
