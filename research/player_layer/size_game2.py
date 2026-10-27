"""
Whole-game player layer, second round of sizing choices on 2022-2025
(folds as in size_game.py; 2026 not read). Compared on the same rows with
size_game.py's wp_game (starter fixed through the 6th, no handedness,
team-level defense).

  python research/player_layer/size_game2.py --states S --hf HF --ctx CTX --prev <size_game out dir> --out O --season Y [--hand] [--align]
  python research/player_layer/size_game2.py --states S --hf HF --ctx CTX --prev P --out O --oof OOF   # summary

Changes (each switchable):
  starter : probability that each team's starter begins inning i, from the
            training folds by (inning now, his batters faced so far); later
            halves mix the starter's and the bullpen's run distributions.
            Always on in this script.
  --hand  : platoon (size_hand.py: league factor ** 0.5 and own splits,
            k = 2000) for the current pitcher and the starters; none for
            the pooled bullpen.
  --align : fielder-level defense: sum of the starting fielders'
            previous-season OAA (HF oaa.parquet) instead of the team's
            in-play record (scale chosen in size_align.py).
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
from size_context import ABBR  # noqa: E402
from size_game import bullpen_counts, game_context, park_table  # noqa: E402
from size_hand import add_hands, split_counts  # noqa: E402
from size_pa import defense_counts  # noqa: E402
from size_runs import game_boot  # noqa: E402
from wp_engine import STATES, Engine, _occ, _state, fit_tables  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
BF_EDGES = [0, 10, 15, 20, 25, 1000]
HAND_BETA, HAND_K = 0.5, 2000.0


def bf_bucket(bf):
    return int(np.searchsorted(BF_EDGES, bf, side="right") - 1)


def starter_table(tr: pd.DataFrame) -> np.ndarray:
    """P(starter begins inning i | starter pitching now in inning c with
    batters-faced bucket b), shape (13, 5, 13); inning index 1..12."""
    first = tr.groupby(["game_pk", "fld_team", "inning"], sort=False).pitcher.first().reset_index()
    first = first.merge(tr.groupby(["game_pk", "fld_team"]).pitcher.first().rename("starter").reset_index(),
                        on=["game_pk", "fld_team"])
    began = {(g, t, i) for g, t, i, p, s in first.itertuples(index=False) if p == s}
    num = np.zeros((13, 5, 13)); den = np.zeros((13, 5))
    rows = tr[tr.cur_is_starter & (tr.inning <= 12)]
    for g, t, c, bf in zip(rows.game_pk, rows.fld_team, rows.inning, rows.bf_before):
        b = bf_bucket(bf)
        den[c, b] += 1
        for i in range(c + 1, 13):
            if (g, t, i) in began:
                num[c, b, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        q = num / den[:, :, None]
    return np.nan_to_num(q)


def run_season(a) -> None:
    t0 = time.time()
    pa = R.lineup_slots(R.load_pa(a.states, SEASONS))
    pa = add_hands(pa, a.ctx)
    bat = R.prior_class_counts(pa, "batter", C).set_index(["batter", "date"])
    pit = R.prior_class_counts(pa, "pitcher", C).set_index(["pitcher", "date"])
    bat = dict(zip(bat.index, bat[C].to_numpy(float)))
    pit = dict(zip(pit.index, pit[C].to_numpy(float)))
    bsp = split_counts(pa, "batter", "phand").set_index(["batter", "phand", "date"])
    psp = split_counts(pa, "pitcher", "bside").set_index(["pitcher", "bside", "date"])
    bsp = dict(zip(bsp.index, bsp[C].to_numpy(float)))
    psp = dict(zip(psp.index, psp[C].to_numpy(float)))
    hands = pd.read_parquet(a.ctx / "hands.parquet").set_index("player")
    bat_side = hands.bat_side.to_dict(); pitch_hand = hands.pitch_hand.to_dict()
    dc = defense_counts(pa)
    dfn = {(t, s, d): v for t, s, d, v in zip(dc.fld_team, dc.season, dc.date, dc[R.BIP].to_numpy(float))}
    pa = game_context(pa)
    done = (pa.cls.ne("OTHER") & pa.pa_complete).astype(int)
    pa["bf_before"] = done.groupby([pa.game_pk, pa.pitcher]).cumsum() - done
    pen = bullpen_counts(pa)
    parks = park_table(a.hf)
    align = None
    if a.align:
        align = json.loads((a.out.parent / "align_scale.json").read_text()) if (a.out.parent / "align_scale.json").exists() else None
        al = pd.concat([pd.read_parquet(a.ctx / f"alignment_{s}.parquet") for s in SEASONS], ignore_index=True)
        oaa = pd.read_parquet(a.hf / "oaa.parquet")
        prev = oaa.groupby(["player_id", "season"]).outs_above_average.sum()
        al["season"] = al.game_pk.map(pa.drop_duplicates("game_pk").set_index("game_pk").season)
        al["oaa_prev"] = [prev.get((p, s - 1), 0) for p, s in zip(al.player, al.season)]
        game_oaa = al.groupby(["game_pk", "team"]).oaa_prev.sum().to_dict()
    pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in SEASONS], ignore_index=True)
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
    Fc = (pd.crosstab(t.combo, t.cls, normalize="index").reindex(columns=C) / L)
    Ftab = {k: v for k, v in zip(Fc.index, Fc.to_numpy())}
    qtab = starter_table(tr)
    d = pa[pa.season == s].copy()
    d["abs_lineup"] = [tuple(np.roll(row, sl)) for row, sl in
                       zip(d[[f"next{k}" for k in range(9)]].to_numpy(), d.slot.to_numpy())]
    zero_bip = np.zeros(len(R.BIP)); zeros = np.zeros(len(C))

    def platoon(ids, pid, date):
        """(9, 7) platoon multipliers of lineup `ids` against pitcher `pid`."""
        ph = pitch_hand.get(pid)
        out = np.ones((len(ids), len(C)))
        if not a.hand or ph not in ("R", "L"):
            return out
        p_all = R.shrunk(pit.get((pid, date), zeros)[None, :], L, 200)[0]
        for k, b in enumerate(ids):
            sd = bat_side.get(b)
            bs = ("L" if ph == "R" else "R") if sd == "S" else sd
            F = Ftab.get(f"{bs}v{ph}")
            if F is None:
                continue
            b_all = R.shrunk(bat.get((b, date), zeros)[None, :], L, 200)[0]
            sp = bsp.get((b, ph, date), zeros)
            b_own = ((sp + HAND_K * b_all * F) / (sp.sum() + HAND_K)) / b_all / F
            pp = psp.get((pid, bs, date), zeros)
            p_own = ((pp + HAND_K * p_all * F) / (pp.sum() + HAND_K)) / p_all / F
            out[k] = F ** HAND_BETA * b_own * p_own
        return out

    def defense(fld, g, season, date):
        if a.align and align is not None:
            o = game_oaa.get((g, fld), 0.0)
            lb = L[[C.index(c) for c in R.BIP]] / L[[C.index(c) for c in R.BIP]].sum()
            # expressed as pseudo in-play counts: fewer hits for positive OAA (scale from size_align.py)
            hit = 1 - align["gamma"] * o / 100.0
            v = lb * np.array([hit, hit, hit, 1.0]); v = v / v.sum()
            return v * 1e6                                  # huge count = the multiplier itself
        return dfn.get((fld, season, date), zero_bip)

    def mult(ids, pid, pcounts, fld, g, season, date, park, plat=True):
        m = Ly.multipliers(np.vstack([bat.get((b, date), zeros) for b in ids]), pcounts[None, :],
                           defense(fld, g, season, date)[None, :], L, park=park)
        return m * platoon(ids, pid, date) if plat else m

    out = np.empty(len(d))
    keys = ["game_pk", "inning", "half", "pitcher", "abs_lineup", "fld_lineup", "fld_next", "bat_pitcher",
            "cur_is_starter", "bat_pitcher_is_starter", "bf_before", "bat_pitcher_bf"]
    seg = d.groupby(keys, sort=False).indices
    col = {c: d[c].to_numpy() for c in ("sidx", "slot", "inning", "diff", "scheduled_innings", "date",
                                         "bat_team", "fld_team", "home_team")}
    n_seg = 0
    for key, rows in seg.items():
        g, inn, half, pitcher, bl, fl, fnext, bpitch, cur_st, bp_st, bf, y_bf = key
        r0 = rows[0]
        date, bt, ft, sched = col["date"][r0], col["bat_team"][r0], col["fld_team"][r0], int(col["scheduled_innings"][r0])
        hab = ABBR.get(int(col["home_team"][r0]))
        park = None if (s == 2025 and hab in ("OAK", "TB")) else parks.get((hab, s))
        pen_f, pen_b = pen.get((ft, s, date), zeros), pen.get((bt, s, date), zeros)
        dist_now = Ly.run_dist(ct, half, mult(bl, pitcher, pit.get((pitcher, date), zeros), ft, g, s, date, park))
        other = "bottom" if half == "top" else "top"
        X_pen = Ly.run_dist(ct, half, mult(bl, None, pen_f, ft, g, s, date, park, plat=False))
        Y_vs_bp = Ly.run_dist(ct, other, mult(fl, bpitch, pit.get((bpitch, date), zeros), bt, g, s, date, park))
        Y_pen = Ly.run_dist(ct, other, mult(fl, None, pen_b, bt, g, s, date, park, plat=False))
        first_y = (inn, "bottom") if half == "top" else (inn + 1, "top")
        qx = qtab[min(inn, 12), bf_bucket(bf)] if cur_st else np.zeros(13)
        y_inn = inn if half == "bottom" else inn - 1            # last inning the batting team pitched (0 = not yet)
        if not bp_st:
            qy = np.zeros(13)
        elif y_inn >= 1:
            qy = qtab[min(y_inn, 12), bf_bucket(y_bf)]
        else:                                                   # top of the 1st: their starter begins the 1st
            qy = qtab[1, 0].copy(); qy[1] = 1.0

        def start_dist(i2, h2, st):
            if h2 == half:
                q = qx[i2] if i2 < 13 else 0.0
                dd = q * dist_now + (1 - q) * X_pen
                return dd[:, st].mean(axis=0)
            q = qy[i2] if i2 < 13 else 0.0
            dd = q * Y_vs_bp + (1 - q) * Y_pen
            return dd[fnext, st] if (i2, h2) == first_y else dd[:, st].mean(axis=0)

        H = G.game_wp(start_dist, sched)
        for i in rows:
            st, slt, inn_i, diff = col["sidx"][i], col["slot"][i], int(col["inning"][i]), int(col["diff"][i])
            dn = np.clip(dist_now[slt, st], 0, None); dn /= dn.sum()
            out[i] = G.wp_now(dn, inn_i, half, diff, sched, H)
        n_seg += 1
        if n_seg % 10000 == 0:
            print("segments", n_seg, round(time.time() - t0), "s", flush=True)
    d["wp_new"] = out
    tag = {(0, 0): "starter", (1, 0): "hand", (0, 1): "align", (1, 1): "handalign"}[(int(a.hand), int(a.align))]
    d[["game_pk", "at_bat_index", "season", "wp_new"]].to_parquet(a.out / f"game2_{tag}_{s}.parquet", index=False)
    print("season", s, "segments", n_seg, round(time.time() - t0), "s", flush=True)


def summarize(a) -> None:
    prev = pd.concat([pd.read_parquet(a.prev / f"game_{s}.parquet") for s in SCORE], ignore_index=True)
    res = {}
    rng = np.random.default_rng(20261027)
    y = prev.home_won.to_numpy(float); g = prev.game_pk.to_numpy()
    se = lambda p: (np.clip(p, 1e-4, 1 - 1e-4) - y) ** 2
    base = {"league": se(prev.wp_league.to_numpy()), "game_v1": se(prev.wp_game.to_numpy())}
    for c in ("league", "game_v1"):
        res[c] = float(base[c].mean())
    for tag in sorted({p.name.split("_", 1)[1].rsplit("_", 1)[0] for p in a.out.glob("game2_*_2022.parquet")}):
        files = [a.out / f"game2_{tag}_{s}.parquet" for s in SCORE]
        if not all(f.exists() for f in files):
            continue
        new = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        m = prev.merge(new, on=["game_pk", "at_bat_index", "season"], how="left")
        sn = se(m.wp_new.to_numpy())
        p = np.clip(m.wp_new.to_numpy(), 1e-4, 1 - 1e-4); band = (p >= .3) & (p < .5)
        res[tag] = {"brier": float(sn.mean()), "band_0.3_0.5_bias_pt": float(100 * (p[band].mean() - y[band].mean())),
                    "minus_league": {"mean": float((sn - base["league"]).mean()), "ci95": game_boot(sn - base["league"], g, rng)},
                    "minus_game_v1": {"mean": float((sn - base["game_v1"]).mean()), "ci95": game_boot(sn - base["game_v1"], g, rng)},
                    "by_season_minus_league": {int(s): float((sn - base["league"])[m.season.to_numpy() == s].mean()) for s in SCORE}}
    (a.out / "summary_game2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--prev", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oof", type=Path)
    ap.add_argument("--season", type=int)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--align", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.season is None:
        summarize(a)
    else:
        run_season(a)


if __name__ == "__main__":
    main()
