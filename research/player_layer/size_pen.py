"""
Late-inning bullpen by usage instead of the pooled bullpen (sizing on
2022-2025; folds as in size_game.py; 2026 not read).

Baseline on the same rows: size_game2.py "starter" (starter-length model,
no handedness, no decay). Only the bullpen part of the later halves
changes; innings 1-6 keep the pooled bullpen.

  python research/player_layer/size_pen.py --states S --hf HF --ctx CTX --prev G2 --out O --season Y --variant A|B
  python research/player_layer/size_pen.py --states S --hf HF --ctx CTX --prev G2 --out O            # summary

Who pitches inning i (7, 8, 9; 10+ uses 7-9 together) for a team, from
data before the game only:
  usage   : each reliever's share of the team's relief plate appearances in
            that inning over the previous 30 days (same season);
  A       : weights = that share;
  B       : weights by the fielding team's lead at the start of the half
            (behind / tied / ahead by 1-3 / ahead by 4+): the share in that
            lead state plus TAU pseudo plate appearances of the inning share;
  available: not used earlier in this game (including the current
            pitcher: he is not assumed to continue), and not on both of the
            previous two days;
  pool    : the mixture is shrunk to the pooled bullpen with weight
            KAPPA / (KAPPA + usage PAs of the available relievers).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import game as G  # noqa: E402
import game_pen as GP  # noqa: E402
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from size_context import ABBR  # noqa: E402
from size_game import bullpen_counts, game_context, park_table  # noqa: E402
from size_game2 import bf_bucket, starter_table  # noqa: E402
from size_pa import defense_counts  # noqa: E402
from size_runs import game_boot  # noqa: E402
from wp_engine import STATES, Engine, _occ, _state, fit_tables  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
WINDOW_DAYS, TAU, KAPPA = 30, 20.0, 30.0
LATE = (7, 8, 9)
LEADS = ("behind", "tied", "close", "big")
S_EMPTY, S_RUNNER2 = G.S_EMPTY, G.S_RUNNER2


def lead_state(lead: int) -> int:
    return 0 if lead < 0 else 1 if lead == 0 else 2 if lead <= 3 else 3


LEAD_OF_D = {sign: np.array([lead_state(int(sign * d)) for d in G.D]) for sign in (1, -1)}


def usage_tables(pa: pd.DataFrame) -> dict:
    """(team, date) -> {pitcher: array (3 innings, 4 lead states) of relief
    PAs over the previous WINDOW_DAYS days of the same season}."""
    r = pa[pa.is_relief & pa.inning.isin(LATE)].copy()
    first = pa.groupby(["game_pk", "half", "inning"], sort=False)[["home_score", "away_score"]].first()
    r = r.join(first.rename(columns={"home_score": "h0", "away_score": "a0"}), on=["game_pk", "half", "inning"])
    lead = np.where(r.half == "top", r.h0 - r.a0, r.a0 - r.h0)     # the fielding team's lead at the half's start
    r["ls"] = [lead_state(int(x)) for x in lead]
    r["ii"] = r.inning - 7
    daily = r.groupby(["fld_team", "season", "date", "pitcher", "ii", "ls"]).size().rename("n").reset_index()
    out = {}
    q = pa[["season", "date"]].drop_duplicates()
    for (t, s), dd in daily.groupby(["fld_team", "season"]):
        dd = dd.sort_values("date")
        dates = dd.date.to_numpy()
        for dt in sorted(q[q.season == s].date):
            lo = np.searchsorted(dates, np.datetime64(dt - pd.Timedelta(days=WINDOW_DAYS)), side="left")
            hi = np.searchsorted(dates, np.datetime64(dt), side="left")
            if hi <= lo:
                continue
            w = dd.iloc[lo:hi]
            u = {}
            for p, ii, ls, n in zip(w.pitcher, w.ii, w.ls, w.n):
                u.setdefault(int(p), np.zeros((3, 4)))[ii, ls] += n
            out[(t, dt)] = u
    return out


def tired_table(pa: pd.DataFrame) -> dict:
    """(team, date) -> relievers who pitched on both of the previous two days."""
    app = pa[pa.is_relief].groupby(["fld_team", "date"]).pitcher.agg(lambda x: set(int(v) for v in x))
    app = app.to_dict()
    out = {}
    for (t, dt) in {(t, d) for t, d in zip(pa.fld_team, pa.date)} | {(t, d) for t, d in zip(pa.bat_team, pa.date)}:
        a1 = app.get((t, dt - pd.Timedelta(days=1)), set())
        a2 = app.get((t, dt - pd.Timedelta(days=2)), set())
        out[(t, dt)] = a1 & a2
    return out


def used_before(pa: pd.DataFrame) -> tuple[list, list]:
    """Per row: pitchers the fielding / batting team used in this game in
    earlier rows plus the current row's pitcher for the fielding team."""
    used_f, used_b = [], []
    cur_g, used = None, None
    for g, ft, bt, p in zip(pa.game_pk.to_numpy(), pa.fld_team.to_numpy(), pa.bat_team.to_numpy(), pa.pitcher.to_numpy()):
        if g != cur_g:
            cur_g, used = g, defaultdict(set)
        used[ft].add(int(p))
        used_f.append(frozenset(used[ft]))
        used_b.append(frozenset(used[bt]))
    return used_f, used_b


def mixture_weights(u: dict, avail: set, variant: str) -> tuple[dict, float]:
    """{(inning index 0-2 or 3 = extra innings, lead state): {pitcher: weight}}
    and the usage PAs of the available relievers (for the pool weight)."""
    W = {}
    tot = float(sum(u[p].sum() for p in avail))
    for ii in range(4):
        rows = {p: (u[p] if ii == 3 else u[p][ii:ii + 1]) for p in avail}
        tot_i = sum(x.sum() for x in rows.values())
        for ls in range(4):
            w = {}
            for p, x in rows.items():
                v = x.sum() if variant == "A" else x[:, ls].sum() + TAU * x.sum() / max(tot_i, 1.0)
                if v > 0:
                    w[p] = v
            W[(ii, ls)] = w
    return W, tot


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
    done = (pa.cls.ne("OTHER") & pa.pa_complete).astype(int)
    pa["bf_before"] = done.groupby([pa.game_pk, pa.pitcher]).cumsum() - done
    pen = bullpen_counts(pa)
    parks = park_table(a.hf)
    pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in SEASONS], ignore_index=True)
    pa["sidx"] = [STATES.index(_state(o, _occ(x), _occ(y), _occ(z)))
                  for o, x, y, z in pa[["outs", "on_1b", "on_2b", "on_3b"]].itertuples(index=False)]
    pa["diff"] = pa.home_score - pa.away_score
    s = a.season
    usage = usage_tables(pa[pa.season == s])
    tired = tired_table(pa[pa.season == s])
    print("prepared", round(time.time() - t0), "s", flush=True)

    tr = pa[pa.season != s]
    eng = Engine(fit_tables(tr, pitch[pitch.game_pk.isin(tr.game_pk)]))
    ct = Ly.class_tables(eng, tr)
    t = tr[tr.cls.isin(C) & tr.pa_complete]
    L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    qtab = starter_table(tr)
    d = pa[pa.season == s].copy()
    if a.games:
        d = d[d.game_pk.isin(pd.unique(d.game_pk)[:a.games])].copy()
    d["used_f"], d["used_b"] = used_before(d)
    d["bf_b"] = [bf_bucket(x) for x in d.bf_before]
    d["ybf_b"] = [bf_bucket(x) for x in d.bat_pitcher_bf]
    d["abs_lineup"] = [tuple(np.roll(row, sl)) for row, sl in
                       zip(d[[f"next{k}" for k in range(9)]].to_numpy(), d.slot.to_numpy())]
    zero_bip = np.zeros(len(R.BIP)); zeros = np.zeros(len(C))

    def mult(ids, pcounts, fld, season, date, park):
        return Ly.multipliers(np.vstack([bat.get((b, date), zeros) for b in ids]), pcounts[None, :],
                              dfn.get((fld, season, date), zero_bip)[None, :], L, park=park)

    variants = list(a.variant)
    outs = {v: np.empty(len(d)) for v in variants}
    keys = ["game_pk", "inning", "half", "pitcher", "abs_lineup", "fld_lineup", "fld_next", "bat_pitcher",
            "cur_is_starter", "bat_pitcher_is_starter", "bf_b", "ybf_b", "used_f", "used_b"]
    seg = d.groupby(keys, sort=False).indices
    col = {c: d[c].to_numpy() for c in ("sidx", "slot", "inning", "diff", "scheduled_innings", "date",
                                         "bat_team", "fld_team", "home_team")}
    n_seg, cache_g, cache = 0, None, {}
    for key, rows in seg.items():
        g, inn, half, pitcher, bl, fl, fnext, bpitch, cur_st, bp_st, bfb, ybfb, used_f, used_b = key
        if g != cache_g:
            cache_g, cache = g, {}
        r0 = rows[0]
        date, bt, ft, sched = col["date"][r0], col["bat_team"][r0], col["fld_team"][r0], int(col["scheduled_innings"][r0])
        hab = ABBR.get(int(col["home_team"][r0]))
        park = None if (s == 2025 and hab in ("OAK", "TB")) else parks.get((hab, s))
        pen_f, pen_b = pen.get((ft, s, date), zeros), pen.get((bt, s, date), zeros)

        def rd(h, ids, pid, pc, fld):
            k = (h, ids, pid, fld)
            if k not in cache:
                cache[k] = Ly.run_dist(ct, h, mult(ids, pc, fld, s, date, park))
            return cache[k]
        other = "bottom" if half == "top" else "top"
        dist_now = rd(half, bl, ("cur", pitcher), pit.get((pitcher, date), zeros), ft)
        X_pen = rd(half, bl, "pool", pen_f, ft)
        Y_vs_bp = rd(other, fl, ("cur", bpitch), pit.get((bpitch, date), zeros), bt)
        Y_pen = rd(other, fl, "pool", pen_b, bt)
        first_y = (inn, "bottom") if half == "top" else (inn + 1, "top")

        def late(team, used, h, ids, fld_team_pen, slot_pick, variant):
            """{(inning idx, lead state): (R,) distribution at each start state} for this team's pitching."""
            u = usage.get((team, date), {})
            avail = {p for p in u if p not in used and p not in tired.get((team, date), set())}
            W, tot = mixture_weights(u, avail, variant)
            pool_w = 1.0 if variant == "N" else KAPPA / (KAPPA + tot)
            pool = rd(h, ids, "pool", fld_team_pen, team)
            res = {}
            for (ii, ls), w in W.items():
                sw = sum(w.values())
                for st in (S_EMPTY, S_RUNNER2):
                    base = slot_pick(pool, st)
                    if sw <= 0 or pool_w >= 1.0:
                        res[(ii, ls, st)] = base
                        continue
                    mix = sum(x * slot_pick(rd(h, ids, ("rp", p), pit.get((p, date), zeros), team), st) for p, x in w.items()) / sw
                    res[(ii, ls, st)] = pool_w * base + (1 - pool_w) * mix
            return res
        avg = lambda dd, st: dd[:, st].mean(axis=0)
        nxt = lambda dd, st: dd[fnext, st]
        qx = qtab[min(inn, 12), bfb] if cur_st else np.zeros(13)
        y_inn = inn if half == "bottom" else inn - 1
        if not bp_st:
            qy = np.zeros(13)
        elif y_inn >= 1:
            qy = qtab[min(y_inn, 12), ybfb]
        else:
            qy = qtab[1, 0].copy(); qy[1] = 1.0

        def pen_dist(tab, i2, h2, st):
            """(ND, R): the bullpen's distribution for every start difference."""
            ii = 3 if i2 >= 10 else i2 - 7
            sign = 1 if h2 == "top" else -1            # fielding team = home in the top half: lead = D
            return np.stack([tab[(ii, ls, st)] for ls in LEAD_OF_D[sign]])

        def start_dist(i2, h2, st):
            if h2 == half:
                q = qx[i2] if i2 < 13 else 0.0
                pen_part = avg(X_pen, st) if i2 < 7 else pen_dist(Xl, i2, h2, st)
                return q * avg(dist_now, st) + (1 - q) * pen_part
            q = qy[i2] if i2 < 13 else 0.0
            if (i2, h2) == first_y:
                pen_part = nxt(Y_pen, st) if i2 < 7 else pen_dist(Yl_first, i2, h2, st)
                return q * nxt(Y_vs_bp, st) + (1 - q) * pen_part
            pen_part = avg(Y_pen, st) if i2 < 7 else pen_dist(Yl_avg, i2, h2, st)
            return q * avg(Y_vs_bp, st) + (1 - q) * pen_part

        for v in variants:
            Xl = late(ft, used_f, half, bl, pen_f, avg, v)
            Yl_avg = late(bt, used_b, other, fl, pen_b, avg, v)
            Yl_first = late(bt, used_b, other, fl, pen_b, nxt, v) if first_y[0] >= 7 else None
            H = GP.game_wp(start_dist, sched)
            for i in rows:
                st, slt, inn_i, diff = col["sidx"][i], col["slot"][i], int(col["inning"][i]), int(col["diff"][i])
                dn = np.clip(dist_now[slt, st], 0, None); dn /= dn.sum()
                outs[v][i] = GP.wp_now(dn, inn_i, half, diff, sched, H)
        n_seg += 1
        if n_seg % 10000 == 0:
            print("segments", n_seg, round(time.time() - t0), "s", flush=True)
    for v in variants:
        d["wp_new"] = outs[v]
        d[["game_pk", "at_bat_index", "season", "wp_new"]].to_parquet(
            a.out / f"pen{v}_{s}{'_g' + str(a.games) if a.games else ''}.parquet", index=False)
    print("season", s, "segments", n_seg, round(time.time() - t0), "s", flush=True)


def summarize(a) -> None:
    prev = pd.concat([pd.read_parquet(a.game / f"game_{s}.parquet") for s in SCORE], ignore_index=True)
    base2 = pd.concat([pd.read_parquet(a.prev / f"game2_starter_{s}.parquet") for s in SCORE], ignore_index=True)
    m0 = prev.merge(base2.rename(columns={"wp_new": "wp_starter"}), on=["game_pk", "at_bat_index", "season"], how="left")
    y = m0.home_won.to_numpy(float); g = m0.game_pk.to_numpy()
    rng = np.random.default_rng(20261028)
    se = lambda p: (np.clip(p, 1e-4, 1 - 1e-4) - y) ** 2
    late_rows = m0.inning.to_numpy() >= 7 if "inning" in m0 else None
    base = se(m0.wp_starter.to_numpy())
    res = {"starter (baseline)": float(base.mean()), "league": float(se(m0.wp_league.to_numpy()).mean())}
    for v in ("A", "B"):
        files = [a.out / f"pen{v}_{s}.parquet" for s in SCORE]
        if not all(f.exists() for f in files):
            continue
        new = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        m = m0.merge(new, on=["game_pk", "at_bat_index", "season"], how="left")
        sn = se(m.wp_new.to_numpy())
        p = np.clip(m.wp_new.to_numpy(), 1e-4, 1 - 1e-4); band = (p >= .3) & (p < .5)
        r = {"brier": float(sn.mean()), "band_0.3_0.5_bias_pt": float(100 * (p[band].mean() - y[band].mean())),
             "minus_starter": {"mean": float((sn - base).mean()), "ci95": game_boot(sn - base, g, rng)},
             "by_season_minus_starter": {int(s): float((sn - base)[m.season.to_numpy() == s].mean()) for s in SCORE}}
        if late_rows is not None:
            lr = late_rows
            r["innings_7plus_minus_starter"] = {"mean": float((sn - base)[lr].mean()),
                                               "ci95": game_boot((sn - base)[lr], g[lr], rng), "n": int(lr.sum())}
        res[v] = r
    (a.out / "summary_pen.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--ctx", type=Path)
    ap.add_argument("--prev", type=Path, help="size_game2.py output dir (game2_starter_*.parquet)")
    ap.add_argument("--game", type=Path, help="size_game.py output dir (game_*.parquet)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--season", type=int)
    ap.add_argument("--variant", default="A",
                    help="letters of A, B, N; N = pooled bullpen only: must equal size_game2 'starter' (gate)")
    ap.add_argument("--games", type=int, help="only the first N games of the season (gate / timing)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.season is None:
        summarize(a)
    else:
        run_season(a)


if __name__ == "__main__":
    main()
