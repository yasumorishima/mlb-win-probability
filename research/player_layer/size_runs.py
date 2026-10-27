"""
Sizing of stages B/C on 2022-2025 (each season with an engine and class
tables fitted on the other four of 2021-2025; player records causal).
2026 is not read.

  python research/player_layer/size_runs.py --states <dir> --oof <oof_wp.parquet> --out <dir>

For every PA row of the scored seasons:
  league : the fold engine's rest-of-half run distribution (= task 1 baseline)
  player : the same with the lineup / pitcher / defense multipliers
Run metrics (rows where the half can be played out: not bottom of the
last scheduled inning or later): P(>=1 more run this half) log loss and
Brier, E[runs] squared error. WP metrics (every row): Brier, log loss,
0.3-0.5 band bias. Paired game-bootstrap intervals for player - league.
Writes rows.parquet (per row predictions) and summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from size_pa import defense_counts  # noqa: E402
from wp_engine import END, MAX_RUNS, STATES, Engine, _clip, _occ, _state, fit_tables  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
RUNS = np.arange(MAX_RUNS + 1)


def prepare(states: Path) -> pd.DataFrame:
    pa = R.lineup_slots(R.load_pa(states, SEASONS))
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C).set_index([who, "date"])
        setattr(prepare, who, cnt)
    dc = defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    k = ["game_pk", "inning", "half"]
    tot = pa.groupby(k).runs_on_play.transform("sum")
    pa["rest_runs"] = tot - (pa.groupby(k).runs_on_play.cumsum() - pa.runs_on_play)
    pa["sidx"] = [STATES.index(_state(o, _occ(a), _occ(b), _occ(c)))
                  for o, a, b, c in pa[["outs", "on_1b", "on_2b", "on_3b"]].itertuples(index=False)]
    pa["diff"] = pa.home_score - pa.away_score
    return pa


def game_boot(d: np.ndarray, g: np.ndarray, rng, n=2000):
    ug, gi = np.unique(g, return_inverse=True)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=n)
    bs = (w @ np.bincount(gi, weights=d)) / (w @ np.bincount(gi).astype(float))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--season", type=int, help="score one season and write rows_<season>.parquet")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.season is None:
        return summarize(a)
    t0 = time.time()
    pa = prepare(a.states)
    pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in SEASONS], ignore_index=True)
    bat = {k: v for k, v in zip(prepare.batter.index, prepare.batter[C].to_numpy(float))}
    pit = {k: v for k, v in zip(prepare.pitcher.index, prepare.pitcher[C].to_numpy(float))}
    print("prepared", round(time.time() - t0), "s")

    parts = []
    for s in [a.season]:
        tr = pa[pa.season != s]
        eng = Engine(fit_tables(tr, pitch[pitch.game_pk.isin(tr.game_pk)]))
        ct = Ly.class_tables(eng, tr)
        t = tr[tr.cls.isin(C) & tr.pa_complete]
        L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
        d = pa[pa.season == s].copy()
        lineup_cols = [f"next{k}" for k in range(9)]

        @lru_cache(maxsize=None)
        def wp_vec(inning, half, diff, sched):
            return np.array([eng._wp_after(inning, half, END, int(r), diff, sched) for r in RUNS])

        p_league = np.empty(len(d)); p_player = np.empty(len(d))
        e_league = np.empty(len(d)); e_player = np.empty(len(d))
        w_league = np.empty(len(d)); w_player = np.empty(len(d))
        league_runs = {h: np.array([eng.runs[h][st] for st in STATES]) for h in ("top", "bottom")}
        # one solve per (game, half-inning, pitcher, lineup as known)
        d["abs_lineup"] = [tuple(np.roll(row[:9], sl)) for row, sl in
                           zip(d[lineup_cols].to_numpy(), d.slot.to_numpy())]
        seg = d.groupby(["game_pk", "inning", "half", "pitcher", "abs_lineup"], sort=False).indices
        pos = {ix: i for i, ix in enumerate(d.index)}
        darr = d[[f"def_{c}" for c in R.BIP]].to_numpy()
        n_solve = 0
        for (g, inn, half, pitcher, lineup), rows in seg.items():
            r0 = rows[0]
            date = d.date.iat[r0]
            B = np.vstack([bat[(b, date)] for b in lineup])
            P = pit[(pitcher, date)][None, :]
            M = Ly.multipliers(B, P, darr[r0][None, :], L)
            dist = Ly.run_dist(ct, half, M)
            n_solve += 1
            for i in rows:
                st, sl = d.sidx.iat[i], d.slot.iat[i]
                dp, dl = dist[sl, st], league_runs[half][st]
                dp = np.clip(dp, 0, None); dp /= dp.sum()
                p_player[i], p_league[i] = 1 - dp[0], 1 - dl[0]
                e_player[i], e_league[i] = RUNS @ dp, RUNS @ dl
                inn_i, diff, sched = int(d.inning.iat[i]), int(d["diff"].iat[i]), int(d.scheduled_innings.iat[i])
                if half == "bottom" and inn_i >= sched and diff > 0:
                    w_player[i] = w_league[i] = 1.0
                else:
                    v = wp_vec(inn_i, half, _clip(diff), sched)
                    w_player[i], w_league[i] = v @ dp, v @ dl
        d["p1_league"], d["p1_player"] = p_league, p_player
        d["er_league"], d["er_player"] = e_league, e_player
        d["wp_league"], d["wp_player"] = w_league, w_player
        parts.append(d[["game_pk", "at_bat_index", "season", "inning", "half", "scheduled_innings", "home_won",
                        "rest_runs", "p1_league", "p1_player", "er_league", "er_player", "wp_league", "wp_player"]])
        print("season", s, "solves", n_solve, round(time.time() - t0), "s")

    pd.concat(parts, ignore_index=True).to_parquet(a.out / f"rows_{a.season}.parquet", index=False)


def summarize(a) -> None:
    rows = pd.concat([pd.read_parquet(a.out / f"rows_{s}.parquet") for s in SCORE], ignore_index=True)
    oof = pd.read_parquet(a.oof, columns=["game_pk", "at_bat_index", "wp_oof"])
    rows = rows.merge(oof, on=["game_pk", "at_bat_index"], how="left")

    rng = np.random.default_rng(20261027)
    summ = {"n_rows": int(len(rows)),
            "gate_wp_league_vs_task1_oof_maxabs": float(np.nanmax(np.abs(rows.wp_league - rows.wp_oof)))}
    runok = ~((rows.half == "bottom") & (rows.inning >= rows.scheduled_innings))
    r = rows[runok]
    y1 = (r.rest_runs >= 1).to_numpy(float)
    g = r.game_pk.to_numpy()
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    run = {"n": int(len(r))}
    for k in ("league", "player"):
        run[k] = {"p1_logloss": float(ll(r[f"p1_{k}"], y1).mean()), "p1_brier": float(((r[f"p1_{k}"] - y1) ** 2).mean()),
                  "er_mse": float(((r[f"er_{k}"] - r.rest_runs) ** 2).mean()),
                  "mean_p1": float(r[f"p1_{k}"].mean()), "mean_er": float(r[f"er_{k}"].mean())}
    run["observed"] = {"p1": float(y1.mean()), "runs": float(r.rest_runs.mean())}
    for name, f in (("p1_logloss", lambda k: ll(r[f"p1_{k}"].to_numpy(), y1)),
                    ("er_se", lambda k: (r[f"er_{k}"] - r.rest_runs).to_numpy() ** 2)):
        dlt = f("player") - f("league")
        run[f"player_minus_league_{name}"] = {"mean": float(dlt.mean()), "ci95": game_boot(dlt, g, rng)}
    summ["runs"] = run
    y = rows.home_won.to_numpy(float)
    g = rows.game_pk.to_numpy()
    wp = {}
    for k in ("league", "player"):
        p = np.clip(rows[f"wp_{k}"].to_numpy(), 1e-4, 1 - 1e-4)
        band = (p >= 0.3) & (p < 0.5)
        wp[k] = {"brier": float(((p - y) ** 2).mean()), "log_loss": float(ll(p, y).mean()),
                 "band_0.3_0.5_bias_pt": float(100 * (p[band].mean() - y[band].mean())), "band_n": int(band.sum())}
    dlt = (np.clip(rows.wp_player, 1e-4, 1 - 1e-4) - y) ** 2 - (np.clip(rows.wp_league, 1e-4, 1 - 1e-4) - y) ** 2
    wp["player_minus_league_brier"] = {"mean": float(dlt.mean()), "ci95": game_boot(dlt.to_numpy(), g, rng)}
    summ["wp"] = wp
    summ["by_season_p1_logloss_diff"] = {
        int(s): float((ll(x.p1_player.to_numpy(), (x.rest_runs >= 1).to_numpy(float))
                       - ll(x.p1_league.to_numpy(), (x.rest_runs >= 1).to_numpy(float))).mean())
        for s, x in r.groupby("season")}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
