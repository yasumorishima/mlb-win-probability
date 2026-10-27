"""
Pre-registered test of the player layer v1 (PREREG.md in this directory).

  predict (no outcome is read; 4 parts can run in parallel):
    python research/player_layer/test_2026.py predict --part K/4 --states S --hf HF --out W [--dry]
  score (reads outcomes, once):
    python research/player_layer/test_2026.py score --states S --out W [--dry]

Real run: target 2026; engine = engine_tables.json (fitted on 2021-2025),
class tables and league rates from 2021-2025, player / pitcher records
from every earlier date of 2021-2026, ENGINE and TEAM predictions from
research/wp_layer/published/pred_2026.parquet (prediction columns only
until the score step).
Dry run (--dry): target 2025, everything fitted on 2021-2024, no TEAM arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import game as G  # noqa: E402
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from size_context import ABBR  # noqa: E402
from size_game import bullpen_counts, game_context, park_table  # noqa: E402
from size_pa import defense_counts  # noqa: E402
from wp_engine import MAX_RUNS, STATES, Engine, _occ, _state, fit_tables  # noqa: E402

C = R.PA_CLASSES
SEED, N_BOOT = 20261028, 2000
TEAM_FILE = ROOT / "research/wp_layer/published/pred_2026.parquet"
INPUT_MD5 = {   # data the run depends on (checked before anything else)
    "pa_2021.parquet": "10947f7add3b8821ddaa70705b845058",
    "pa_2022.parquet": "bbacd8f9703d5922d615903ebd693ca3",
    "pa_2023.parquet": "c8e4f89027179b1b09465a3d73580011",
    "pa_2024.parquet": "7d7f79769fae15f0c431d6389b4624ea",
    "pa_2025.parquet": "9d82f22f73ddd0189714b5c277a315ef",
    "pa_2026.parquet": "1c54a8d674846aac475aa87ce27fcc3c",
    "park_factors.parquet": "3f1c06008ec1fec5012a8f50338b9687",
    "engine_tables.json": "ffd4b264f96c29434ba21afb4ac7c9fb",
    "pred_2026.parquet": "c9e35870ab157190d2d81b1f26b23b87",
}
NEW_PARKS = {2025: ("OAK", "TB"), 2026: ("OAK", "TB")}   # park factor set to average (venue not in the 3-year window)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def check_inputs(states: Path, hf: Path | None) -> None:
    paths = {f"pa_{s}.parquet": states / f"pa_{s}.parquet" for s in range(2021, 2027)}
    paths["engine_tables.json"] = ROOT / "engine_tables.json"
    paths["pred_2026.parquet"] = TEAM_FILE
    if hf is not None:
        paths["park_factors.parquet"] = hf / "park_factors.parquet"
    bad = {k: md5(p) for k, p in paths.items() if md5(p) != INPUT_MD5[k]}
    if bad:
        sys.exit(f"input md5 mismatch: {bad}")


def predict(a) -> None:
    check_inputs(a.states, a.hf)
    t0 = time.time()
    target = 2025 if a.dry else 2026
    train = list(range(2021, target))
    seasons = train + [target]
    pa = R.lineup_slots(R.load_pa(a.states, seasons))
    bat = R.prior_class_counts(pa, "batter", C).set_index(["batter", "date"])
    pit = R.prior_class_counts(pa, "pitcher", C).set_index(["pitcher", "date"])
    bat = dict(zip(bat.index, bat[C].to_numpy(float)))
    pit = dict(zip(pit.index, pit[C].to_numpy(float)))
    dc = defense_counts(pa)
    dfn = {(t, s, d): v for t, s, d, v in zip(dc.fld_team, dc.season, dc.date, dc[R.BIP].to_numpy(float))}
    pa = game_context(pa)
    pen = bullpen_counts(pa)
    parks = park_table(a.hf)
    pa["sidx"] = [STATES.index(_state(o, _occ(x), _occ(y), _occ(z)))
                  for o, x, y, z in pa[["outs", "on_1b", "on_2b", "on_3b"]].itertuples(index=False)]
    pa["diff"] = pa.home_score - pa.away_score

    tr = pa[pa.season.isin(train)]
    if a.dry:
        pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in train], ignore_index=True)
        eng = Engine(fit_tables(tr, pitch))
    else:
        eng = Engine.load(ROOT / "engine_tables.json")
    ct = Ly.class_tables(eng, tr)
    t = tr[tr.cls.isin(C) & tr.pa_complete]
    L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    d = pa[pa.season == target].copy()
    k, n = (int(x) for x in a.part.split("/"))
    d = d[d.game_pk % n == k].copy()
    d["abs_lineup"] = [tuple(np.roll(row, sl)) for row, sl in
                       zip(d[[f"next{j}" for j in range(9)]].to_numpy(), d.slot.to_numpy())]
    zeros, zero_bip = np.zeros(len(C)), np.zeros(len(R.BIP))
    lg = {h: np.array([eng.runs[h][st] for st in STATES]) for h in ("top", "bottom")}

    def mult(ids, pcounts, fld, date, park):
        return Ly.multipliers(np.vstack([bat.get((b, date), zeros) for b in ids]), pcounts[None, :],
                              dfn.get((fld, target, date), zero_bip)[None, :], L, park=park)

    out = {c: np.full(len(d), np.nan) for c in ("wp_player", "wp_engine", "p1_player", "p1_engine")}
    keys = ["game_pk", "inning", "half", "pitcher", "abs_lineup", "fld_lineup", "fld_next", "bat_pitcher",
            "cur_is_starter", "bat_pitcher_is_starter"]
    col = {c: d[c].to_numpy() for c in ("sidx", "slot", "inning", "diff", "scheduled_innings", "date",
                                         "bat_team", "fld_team", "home_team")}
    Hl_cache = {}
    for key, rows in d.groupby(keys, sort=False).indices.items():
        g, inn, half, pitcher, bl, fl, fnext, bpitch, cur_st, bp_st = key
        r0 = rows[0]
        date, bt, ft, sched = col["date"][r0], col["bat_team"][r0], col["fld_team"][r0], int(col["scheduled_innings"][r0])
        hab = ABBR.get(int(col["home_team"][r0]))
        park = None if hab in NEW_PARKS.get(target, ()) else parks.get((hab, target))
        pen_f, pen_b = pen.get((ft, target, date), zeros), pen.get((bt, target, date), zeros)
        dist_now = Ly.run_dist(ct, half, mult(bl, pit.get((pitcher, date), zeros), ft, date, park))
        other = "bottom" if half == "top" else "top"
        X_pen = Ly.run_dist(ct, half, mult(bl, pen_f, ft, date, park))
        Y_vs_bp = Ly.run_dist(ct, other, mult(fl, pit.get((bpitch, date), zeros), bt, date, park))
        Y_pen = Ly.run_dist(ct, other, mult(fl, pen_b, bt, date, park))
        first_y = (inn, "bottom") if half == "top" else (inn + 1, "top")

        def start_dist(i2, h2, st):
            if h2 == half:
                dd = dist_now if (cur_st and i2 <= G.STARTER_LAST) else X_pen
                return dd[:, st].mean(axis=0)
            dd = Y_vs_bp if (bp_st and i2 <= G.STARTER_LAST) else Y_pen
            return dd[fnext, st] if (i2, h2) == first_y else dd[:, st].mean(axis=0)

        H = G.game_wp(start_dist, sched)
        if sched not in Hl_cache:
            Hl_cache[sched] = G.game_wp(lambda i2, h2, st: lg[h2][st], sched)
        for i in rows:
            st, slt, inn_i, diff = col["sidx"][i], col["slot"][i], int(col["inning"][i]), int(col["diff"][i])
            dn = np.clip(dist_now[slt, st], 0, None); dn /= dn.sum()
            out["wp_player"][i] = G.wp_now(dn, inn_i, half, diff, sched, H)
            out["wp_engine"][i] = G.wp_now(lg[half][st], inn_i, half, diff, sched, Hl_cache[sched])
            out["p1_player"][i] = 1 - dn[0]
            out["p1_engine"][i] = 1 - lg[half][st][0]
    for c, v in out.items():
        d[c] = v
    a.out.mkdir(parents=True, exist_ok=True)
    keep = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings", "diff",
            "wp_player", "wp_engine", "p1_player", "p1_engine"]
    d[keep].to_parquet(a.out / f"pred_{target}_part{k}.parquet", index=False)
    print("part", a.part, "rows", len(d), round(time.time() - t0), "s", flush=True)


def boot(diffs: dict, games: np.ndarray, rng) -> dict:
    """Game bootstrap of mean differences; all arms resampled together."""
    ug, gi = np.unique(games, return_inverse=True)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)
    cnt = w @ np.bincount(gi).astype(float)
    out = {}
    for k, v in diffs.items():
        bs = (w @ np.bincount(gi, weights=v)) / cnt
        out[k] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "p_one_sided": float((1 + (bs >= 0).sum()) / (N_BOOT + 1))}
    return out


def calib(p, y):
    b = np.clip((p * 10).astype(int), 0, 9)
    return [{"bin": k / 10, "n": int((b == k).sum()), "pred": float(p[b == k].mean()) if (b == k).any() else None,
             "obs": float(y[b == k].mean()) if (b == k).any() else None} for k in range(10)]


def verify_frozen(required: bool) -> dict:
    """md5 of every file listed under "Frozen files" in PREREG.md (lines
    `<md5>  <repo-relative path>`); the real run requires Status: FROZEN
    and every md5 to match."""
    text = (HERE / "PREREG.md").read_text()
    listed = re.findall(r"^([0-9a-f]{32})  (\S+)$", text, flags=re.M)
    got = {path: md5(ROOT / path) for _, path in listed}
    bad = {path: got[path] for m, path in listed if got[path] != m}
    frozen = re.search(r"^Status: FROZEN", text, flags=re.M) is not None
    if required and (not frozen or not listed or bad):
        sys.exit(f"frozen check failed: status FROZEN={frozen}, listed={len(listed)}, mismatched={bad}")
    return {"status_frozen": frozen, "n_listed": len(listed), "mismatched": bad,
            "prereg_md5": md5(HERE / "PREREG.md"), "script_md5": md5(Path(__file__))}


def score(a) -> None:
    check_inputs(a.states, a.hf)
    target = 2025 if a.dry else 2026
    res = {"target": target, "frozen": verify_frozen(required=not a.dry),
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    parts = [pd.read_parquet(a.out / f"pred_{target}_part{k}.parquet") for k in range(4)]
    pred = pd.concat(parts, ignore_index=True)
    if pred.duplicated(["game_pk", "at_bat_index"]).any():
        sys.exit("duplicate rows across parts")
    for c in ("wp_player", "wp_engine", "p1_player", "p1_engine"):
        if not np.isfinite(pred[c].to_numpy()).all():
            sys.exit(f"non-finite values in {c}")
    if not a.dry:
        # gate BEFORE any outcome is loaded: the recomputed engine equals the
        # published p_ENGINE where both clip the score the same way (|diff| <= 15)
        team = pd.read_parquet(TEAM_FILE, columns=["game_pk", "at_bat_index", "p_ENGINE", "p_TEAM"])
        pred = pred.merge(team, on=["game_pk", "at_bat_index"], how="left")
        same = pred["diff"].abs().to_numpy() <= 15
        res["gate"] = {"rows": int(len(pred)), "rows_compared": int(same.sum()), "rows_abs_diff_over_15": int((~same).sum()),
                       "maxabs_engine_vs_published": float(np.abs(pred.wp_engine - pred.p_ENGINE)[same].max())}
        if (len(pred) != 184327 or pred.p_TEAM.isna().any() or not np.isfinite(pred.p_TEAM).all()
                or not res["gate"]["maxabs_engine_vs_published"] < 1e-9):
            sys.exit(f"gate failed: {res['gate']}")
    pa = R.load_pa(a.states, [target])
    k = ["game_pk", "inning", "half"]
    tot = pa.groupby(k).runs_on_play.transform("sum")
    pa["rest_runs"] = tot - (pa.groupby(k).runs_on_play.cumsum() - pa.runs_on_play)
    last = pa.groupby(k).tail(1)
    cut = last[(last.outs_after < 3) & ~((last.half == "bottom") & (last.inning >= last.scheduled_innings))]
    cut_keys = set(zip(cut.game_pk, cut.inning, cut.half))       # halves ended early (called / suspended)
    df = pred.merge(pa[["game_pk", "at_bat_index", "home_won", "rest_runs"]], on=["game_pk", "at_bat_index"], how="left")
    res.update({"n_rows": int(len(df)), "n_games": int(df.game_pk.nunique())})
    arms = {"PLAYER": df.wp_player.to_numpy(), "ENGINE": df.wp_engine.to_numpy()}
    if not a.dry:
        arms["TEAM"] = df.p_TEAM.to_numpy()
    y = df.home_won.to_numpy(float)
    g = df.game_pk.to_numpy()
    se = {kk: (v - y) ** 2 for kk, v in arms.items()}                 # Brier on the unclipped probabilities
    res["arms"] = {}
    for kk, v in arms.items():
        p = np.clip(v, 1e-4, 1 - 1e-4)                                # clipped for log loss only
        band = (v >= 0.3) & (v < 0.5)
        res["arms"][kk] = {"brier": float(se[kk].mean()), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                           "band_0.3_0.5_gap_pt": float(100 * (v[band].mean() - y[band].mean())), "band_n": int(band.sum()),
                           "calibration": calib(v, y)}
    # H3 rows: the half can be played out, and it was not ended early
    ok = ~((df.half == "bottom") & (df.inning >= df.scheduled_innings)).to_numpy()
    ok &= np.array([(gg, ii, hh) not in cut_keys for gg, ii, hh in zip(df.game_pk, df.inning, df.half)])
    y1 = (df.rest_runs.to_numpy()[ok] >= 1).astype(float)
    ll = lambda p: -(y1 * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y1) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)))
    l_pl, l_en = ll(df.p1_player.to_numpy()[ok]), ll(df.p1_engine.to_numpy()[ok])
    res["runs"] = {"n": int(ok.sum()), "halves_ended_early": len(cut_keys), "observed_p1": float(y1.mean()),
                   "PLAYER_logloss": float(l_pl.mean()), "ENGINE_logloss": float(l_en.mean())}
    rng = np.random.default_rng(SEED)
    diffs = {"H1_PLAYER_minus_ENGINE_brier": se["PLAYER"] - se["ENGINE"]}
    if not a.dry:
        diffs["H2_PLAYER_minus_TEAM_brier"] = se["PLAYER"] - se["TEAM"]
    b = boot(diffs, g, rng)
    b.update(boot({"H3_PLAYER_minus_ENGINE_p1_logloss": l_pl - l_en}, g[ok], rng))
    res["tests"] = b
    pv = {kk: v["p_one_sided"] for kk, v in b.items()}
    order = sorted(pv, key=pv.get)
    holm, prev = {}, 0.0
    for i, h in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - i) * pv[h]))
        holm[h] = prev
    res["holm_adjusted_p"] = holm
    res["supported"] = {h: bool(v <= 0.05) for h, v in holm.items()}
    # reported only
    stage = pd.cut(df.inning, [0, 3, 6, 100], labels=["1-3", "4-6", "7+"])
    res["brier_by_stage"] = {str(s): {kk: float(se[kk][(stage == s).to_numpy()].mean()) for kk in arms} for s in stage.cat.categories}
    (a.out / f"results_{target}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({kk: res[kk] for kk in res if kk not in ("arms",)}, indent=1))
    for kk, v in res["arms"].items():
        print(kk, {m: round(v[m], 6) for m in ("brier", "log_loss", "band_0.3_0.5_gap_pt")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "score"])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", help="K/4, required for predict")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.mode == "predict":
        if a.hf is None or a.part is None or not a.part.endswith("/4"):
            sys.exit("predict needs --hf and --part K/4")
        predict(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
