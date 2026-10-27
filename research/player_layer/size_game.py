"""
Sizing of the whole-game player layer on 2022-2025 (each season with an
engine and class tables fitted on the other four of 2021-2025; records
causal; 2026 not read).

  python research/player_layer/size_game.py --states <dir> --hf <dir> --out <dir> --season S
  python research/player_layer/size_game.py --states <dir> --hf <dir> --out <dir>      # summary

Per PA row it writes three home WPs:
  wp_league  the fold engine (= task 1 out-of-fold WP)
  wp_half    player layer in the current half only (later halves league)
  wp_game    player layer in every remaining half (game.py), with park
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import game as G  # noqa: E402
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from size_context import ABBR, PF_COLS  # noqa: E402
from size_pa import defense_counts  # noqa: E402
from size_runs import game_boot  # noqa: E402
from wp_engine import END, MAX_RUNS, STATES, Engine, _clip, _occ, _state, fit_tables  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
RUNS = np.arange(MAX_RUNS + 1)


def game_context(pa: pd.DataFrame) -> pd.DataFrame:
    """Per row: the fielding team's lineup as known (absolute slots) and
    its next slot; whether the current pitcher is the fielding team's
    starter; the batting team's latest pitcher and whether he is its
    starter (the pitcher it will use in the next opposite half)."""
    pa = pa.sort_values(["game_pk", "at_bat_index"]).copy()
    first_occ = {}
    starter = {}
    for (g, t, slot), b in pa.groupby(["game_pk", "bat_team", "slot"], sort=False).batter.first().items():
        first_occ.setdefault((g, t), {})[slot] = b
    for (g, t), p in pa.groupby(["game_pk", "fld_team"], sort=False).pitcher.first().items():
        starter[(g, t)] = p
    n = len(pa)
    fld_lineup = np.zeros((n, 9), dtype=np.int64)
    fld_next = np.zeros(n, dtype=np.int64)
    bat_pitcher = np.zeros(n, dtype=np.int64)
    bat_pitcher_bf = np.zeros(n, dtype=np.int64)
    gk = pa.game_pk.to_numpy(); bt = pa.bat_team.to_numpy(); ft = pa.fld_team.to_numpy()
    sl = pa.slot.to_numpy(); bb = pa.batter.to_numpy(); pp = pa.pitcher.to_numpy()
    done = (pa.cls.ne("OTHER") & pa.pa_complete).to_numpy()
    cur_g = None
    for i in range(n):
        g = gk[i]
        if g != cur_g:
            cur_g = g
            lineup, nxt, lastp, bfc = {}, {}, {}, {}
        b_t, f_t = bt[i], ft[i]
        if f_t not in lineup:
            lineup[f_t] = dict(first_occ.get((g, f_t), {}))
            nxt[f_t] = 0
        if b_t not in lineup:
            lineup[b_t] = dict(first_occ.get((g, b_t), {}))
            nxt[b_t] = 0
        lineup[b_t][sl[i]] = bb[i]
        lastp[f_t] = pp[i]
        fld_lineup[i] = [lineup[f_t].get(k, -1) for k in range(9)]
        fld_next[i] = nxt[f_t]
        bat_pitcher[i] = lastp.get(b_t, starter.get((g, b_t), -1))
        bat_pitcher_bf[i] = bfc.get(bat_pitcher[i], 0)      # his batters faced so far (none after this row)
        if done[i]:
            nxt[b_t] = (sl[i] + 1) % 9
            bfc[pp[i]] = bfc.get(pp[i], 0) + 1
    pa["fld_lineup"] = [tuple(x) for x in fld_lineup]
    pa["fld_next"] = fld_next
    pa["bat_pitcher"] = bat_pitcher
    pa["bat_pitcher_bf"] = bat_pitcher_bf
    pa["cur_is_starter"] = [p == starter.get((g, t)) for g, t, p in zip(gk, ft, pp)]
    pa["bat_pitcher_is_starter"] = [p == starter.get((g, t)) for g, t, p in zip(gk, bt, bat_pitcher)]
    pa["is_relief"] = ~pa.cur_is_starter
    return pa


def bullpen_counts(pa: pd.DataFrame) -> dict:
    """(team, season, date) -> class counts of the team's relievers on
    earlier dates of the season."""
    rows = pa[pa.is_relief & pa.cls.isin(C) & pa.pa_complete]
    d = pd.crosstab([rows.fld_team, rows.season, rows.date], rows.cls).reindex(columns=C, fill_value=0).sort_index()
    thru = d.groupby(level=[0, 1]).cumsum().reset_index()
    thru["date"] = thru["date"] + pd.Timedelta(days=1)
    q = pa[["fld_team", "season", "date"]].drop_duplicates().sort_values("date")
    q2 = q.rename(columns={"fld_team": "team"})
    q2 = pd.concat([q2, pa[["bat_team", "season", "date"]].drop_duplicates().rename(columns={"bat_team": "team"})]).drop_duplicates().sort_values("date")
    m = pd.merge_asof(q2, thru.rename(columns={"fld_team": "team"}).sort_values("date"), on="date",
                      by=["team", "season"], direction="backward")
    m[C] = m[C].fillna(0).astype(float)
    return {(t, s, dt): v for t, s, dt, v in zip(m.team, m.season, m.date, m[C].to_numpy(float))}


def park_table(hf: Path) -> dict:
    pf = pd.read_parquet(hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    out = {}
    for r in pf.itertuples():
        v = np.ones(len(C))
        for c, col in PF_COLS.items():
            x = getattr(r, col)
            v[C.index(c)] = 1.0 if pd.isna(x) else x / 100.0      # missing factor = average
        out[(r.team, r.season + 1)] = v                       # used in the following season
    return out


def run_season(a) -> None:
    t0 = time.time()
    pa = R.lineup_slots(R.load_pa(a.states, SEASONS))
    bat = R.prior_class_counts(pa, "batter", C).set_index(["batter", "date"])
    pit = R.prior_class_counts(pa, "pitcher", C).set_index(["pitcher", "date"])
    bat = dict(zip(bat.index, bat[C].to_numpy(float)))
    pit = dict(zip(pit.index, pit[C].to_numpy(float)))
    dc = defense_counts(pa)
    dfn = {(t, s, d): v for t, s, d, v in zip(dc.fld_team, dc.season, dc.date, dc[R.BIP].to_numpy(float))}
    pa = game_context(pa)
    pen = bullpen_counts(pa)
    parks = park_table(a.hf)
    pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in SEASONS], ignore_index=True)
    k = ["game_pk", "inning", "half"]
    pa["sidx"] = [STATES.index(_state(o, _occ(x), _occ(y), _occ(z)))
                  for o, x, y, z in pa[["outs", "on_1b", "on_2b", "on_3b"]].itertuples(index=False)]
    pa["diff"] = pa.home_score - pa.away_score
    print("prepared", round(time.time() - t0), "s", flush=True)

    s = a.season
    tr = pa[pa.season != s]
    eng = Engine(fit_tables(tr, pitch[pitch.game_pk.isin(tr.game_pk)]))
    ct = Ly.class_tables(eng, tr)
    t = tr[tr.cls.isin(C) & tr.pa_complete]
    L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    d = pa[pa.season == s].copy()
    d["abs_lineup"] = [tuple(np.roll(row, sl)) for row, sl in
                       zip(d[[f"next{k}" for k in range(9)]].to_numpy(), d.slot.to_numpy())]
    zero_bip = np.zeros(len(R.BIP))
    lg = {h: np.array([eng.runs[h][st] for st in STATES]) for h in ("top", "bottom")}

    def bcounts(ids, date):
        return np.vstack([bat.get((b, date), np.zeros(len(C))) for b in ids])

    def mult(ids, pcounts, fld, season, date, park):
        return Ly.multipliers(bcounts(ids, date), pcounts[None, :], dfn.get((fld, season, date), zero_bip)[None, :],
                              L, park=park)

    out = {c: np.empty(len(d)) for c in ("wp_league", "wp_half", "wp_game")}
    keys = ["game_pk", "inning", "half", "pitcher", "abs_lineup", "fld_lineup", "fld_next", "bat_pitcher",
            "cur_is_starter", "bat_pitcher_is_starter"]
    seg = d.groupby(keys, sort=False).indices
    col = {c: d[c].to_numpy() for c in ("sidx", "slot", "inning", "diff", "scheduled_innings", "date",
                                         "bat_team", "fld_team", "home_team", "half")}
    n_seg = 0
    Hl_cache = {}
    for key, rows in seg.items():
        g, inn, half, pitcher, bl, fl, fnext, bpitch, cur_st, bp_st = key
        r0 = rows[0]
        date, bt, ft, sched = col["date"][r0], col["bat_team"][r0], col["fld_team"][r0], int(col["scheduled_innings"][r0])
        park = parks.get((ABBR.get(int(col["home_team"][r0])), s))
        if park is not None and s == 2025 and ABBR.get(int(col["home_team"][r0])) in ("OAK", "TB"):
            park = None
        zeros = np.zeros(len(C))
        p_cur = pit.get((pitcher, date), zeros)
        p_bat = pit.get((bpitch, date), zeros)
        pen_f, pen_b = pen.get((ft, s, date), zeros), pen.get((bt, s, date), zeros)
        # current half: player multipliers without park (wp_half, comparable to size_runs) and with park
        dist_half = Ly.run_dist(ct, half, mult(bl, p_cur, ft, s, date, None))
        dist_now = Ly.run_dist(ct, half, mult(bl, p_cur, ft, s, date, park))
        other = "bottom" if half == "top" else "top"
        X_pen = Ly.run_dist(ct, half, mult(bl, pen_f, ft, s, date, park))
        Y_vs_bp = Ly.run_dist(ct, other, mult(fl, p_bat, bt, s, date, park))
        Y_pen = Ly.run_dist(ct, other, mult(fl, pen_b, bt, s, date, park))
        first_y = (inn, "bottom") if half == "top" else (inn + 1, "top")

        def start_dist(i2, h2, st):
            if h2 == half:                                  # batting team's later halves
                dd = dist_now if (cur_st and i2 <= G.STARTER_LAST) else X_pen
                return dd[:, st].mean(axis=0)
            dd = Y_vs_bp if (bp_st and i2 <= G.STARTER_LAST) else Y_pen
            if (i2, h2) == first_y:
                return dd[fnext, st]
            return dd[:, st].mean(axis=0)

        H = G.game_wp(start_dist, sched)
        if sched not in Hl_cache:
            Hl_cache[sched] = G.game_wp(lambda i2, h2, st: lg[h2][st], sched)
        Hl = Hl_cache[sched]
        for i in rows:
            st, slt, inn_i, diff = col["sidx"][i], col["slot"][i], int(col["inning"][i]), int(col["diff"][i])
            dn = np.clip(dist_now[slt, st], 0, None); dn /= dn.sum()
            dh = np.clip(dist_half[slt, st], 0, None); dh /= dh.sum()
            out["wp_game"][i] = G.wp_now(dn, inn_i, half, diff, sched, H)
            out["wp_half"][i] = G.wp_now(dh, inn_i, half, diff, sched, Hl)
            out["wp_league"][i] = G.wp_now(lg[half][st], inn_i, half, diff, sched, Hl)
        n_seg += 1
        if n_seg % 10000 == 0:
            print("segments", n_seg, round(time.time() - t0), "s", flush=True)
    for c, v in out.items():
        d[c] = v
    d[["game_pk", "at_bat_index", "season", "home_won", "wp_league", "wp_half", "wp_game"]].to_parquet(
        a.out / f"game_{s}.parquet", index=False)
    print("season", s, "segments", n_seg, round(time.time() - t0), "s", flush=True)


def summarize(a) -> None:
    rows = pd.concat([pd.read_parquet(a.out / f"game_{s}.parquet") for s in SCORE], ignore_index=True)
    oof = pd.read_parquet(a.oof, columns=["game_pk", "at_bat_index", "wp_oof"])
    rows = rows.merge(oof, on=["game_pk", "at_bat_index"], how="left")
    y = rows.home_won.to_numpy(float)
    g = rows.game_pk.to_numpy()
    rng = np.random.default_rng(20261027)
    res = {"n": int(len(rows)), "gate_league_vs_oof_maxabs": float(np.abs(rows.wp_league - rows.wp_oof).max())}
    se = {}
    for c in ("wp_league", "wp_half", "wp_game"):
        p = np.clip(rows[c].to_numpy(), 1e-4, 1 - 1e-4)
        se[c] = (p - y) ** 2
        band = (p >= 0.3) & (p < 0.5)
        res[c] = {"brier": float(se[c].mean()),
                  "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                  "band_0.3_0.5_bias_pt": float(100 * (p[band].mean() - y[band].mean())), "band_n": int(band.sum()),
                  "calibration": [{"bin": k / 10, "n": int(((p * 10).astype(int) == k).sum()),
                                   "pred": float(p[(p * 10).astype(int) == k].mean()),
                                   "obs": float(y[(p * 10).astype(int) == k].mean())} for k in range(10)]}
    for c in ("wp_half", "wp_game"):
        dlt = se[c] - se["wp_league"]
        res[f"{c}_minus_league_brier"] = {"mean": float(dlt.mean()), "ci95": game_boot(dlt, g, rng)}
    dlt = se["wp_game"] - se["wp_half"]
    res["game_minus_half_brier"] = {"mean": float(dlt.mean()), "ci95": game_boot(dlt, g, rng)}
    res["by_season_game_minus_league"] = {int(s): float((se["wp_game"] - se["wp_league"])[rows.season.to_numpy() == s].mean())
                                          for s in SCORE}
    (a.out / "summary_game.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "calibration" not in v}, indent=1))
    for c in ("wp_league", "wp_half", "wp_game"):
        print(c, {k: round(v, 6) if isinstance(v, float) else v for k, v in res[c].items() if k != "calibration"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oof", type=Path)
    ap.add_argument("--season", type=int)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.season is None:
        summarize(a)
    else:
        run_season(a)


if __name__ == "__main__":
    main()
