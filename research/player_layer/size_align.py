"""
Stage A, fielder-level defense: the fielding team's starting fielders in
this game (Stats API boxscore, fetch_context.py) and their
previous-season OAA (HF oaa.parquet, summed over positions). On top of
V5 + park. 2022-2025, league tables from the other four of 2021-2025.
2026 not read.

  python research/player_layer/size_align.py --states S --hf HF --ctx CTX --out <json>

  total : hits on balls in play x (1 - gamma * OAA_total / 100)
  split : singles x (1 - g_if * OAA_infield / 100 - g_of * OAA_outfield / 100),
          doubles / triples x (1 - g_of * OAA_outfield / 100)
Late defensive substitutions are not seen (starting alignment only).
Players without a previous-season OAA row count as 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
import size_pa as S  # noqa: E402
from size_context import ABBR, PF_COLS  # noqa: E402

C = R.PA_CLASSES
INF = {"C", "1B", "2B", "3B", "SS"}


def game_oaa(ctx: Path, hf: Path, seasons) -> pd.DataFrame:
    al = pd.concat([pd.read_parquet(ctx / f"alignment_{s}.parquet").assign(season=s) for s in seasons], ignore_index=True)
    oaa = pd.read_parquet(hf / "oaa.parquet")
    prev = oaa.groupby(["player_id", "season"]).outs_above_average.sum()
    al["oaa_prev"] = [float(prev.get((p, s - 1), 0.0)) for p, s in zip(al.player, al.season)]
    al["has_prev"] = [(p, s - 1) in prev.index for p, s in zip(al.player, al.season)]
    al["grp"] = np.where(al.pos.isin(INF), "if", "of")
    g = al.pivot_table(index=["game_pk", "team"], columns="grp", values="oaa_prev", aggfunc="sum").fillna(0.0)
    g["n_fielders"] = al.groupby(["game_pk", "team"]).size()
    g["n_with_prev"] = al.groupby(["game_pk", "team"]).has_prev.sum()
    return g.reset_index().rename(columns={"if": "oaa_if", "of": "oaa_of", "team": "fld_team"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pa = R.load_pa(a.states, S.SEASONS)
    pa = pa[pa.cls.isin(C) & pa.pa_complete].reset_index(drop=True)
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C)
        pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
    dc = S.defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    go = game_oaa(a.ctx, a.hf, S.SEASONS)
    pa = pa.merge(go, on=["game_pk", "fld_team"], how="left")
    res = {"rows_without_alignment": int(pa.oaa_if.isna().sum()),
           "fielders_per_team_game": pa.drop_duplicates(["game_pk", "fld_team"]).n_fielders.describe().round(2).to_dict(),
           "share_fielders_with_prev_oaa": float(pa.drop_duplicates(["game_pk", "fld_team"]).n_with_prev.sum()
                                                 / pa.drop_duplicates(["game_pk", "fld_team"]).n_fielders.sum())}
    pa[["oaa_if", "oaa_of"]] = pa[["oaa_if", "oaa_of"]].fillna(0.0)
    pa["sk"] = S.state_key(pa)
    pf = pd.read_parquet(a.hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    pf = pf.assign(season=pf.season + 1).set_index(["team", "season"])
    j = pf.reindex(pd.MultiIndex.from_arrays([pa.home_team.map(ABBR), pa.season]))
    PF = np.ones((len(pa), len(C)))
    for c, col in PF_COLS.items():
        PF[:, C.index(c)] = j[col].to_numpy() / 100.0
    PF[((pa.home_team.map(ABBR).isin(["OAK", "TB"])) & (pa.season == 2025)).to_numpy()] = np.nan
    PF[np.isnan(PF)] = 1.0
    y = pa.cls.map({c: i for i, c in enumerate(C)}).to_numpy()
    folds = {}
    for s in S.SCORE:
        st, L = S.league_tables(pa[pa.season != s])
        m = (pa.season == s).to_numpy(); d = pa[m]
        M = Ly.multipliers(d[[f"batter_{c}" for c in C]].to_numpy(), d[[f"pitcher_{c}" for c in C]].to_numpy(),
                           d[[f"def_{c}" for c in R.BIP]].to_numpy(), L)
        folds[s] = (st.reindex(d.sk).to_numpy() * M * PF[m] ** 0.5, d.oaa_if.to_numpy(), d.oaa_of.to_numpy(),
                    y[m], d.game_pk.to_numpy())
    i1, i2, i3 = C.index("1B"), C.index("2B"), C.index("3B")

    def ll(g_if, g_of, per_row=False):
        out = []
        for base, oi, oo, yy, _ in folds.values():
            p = base.copy()
            p[:, i1] *= np.clip(1 - g_if * oi / 100 - g_of * oo / 100, 0.5, 1.5)
            p[:, [i2, i3]] *= np.clip(1 - g_of * oo / 100, 0.5, 1.5)[:, None]
            p = p / p.sum(axis=1, keepdims=True)
            out.append(-np.log(p[np.arange(len(yy)), yy]))
        v = np.concatenate(out)
        return v if per_row else float(v.mean())

    grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    res["grid_total"] = [{"gamma": gm, "ll": ll(gm, gm)} for gm in grid]
    res["grid_split"] = [{"g_if": gi, "g_of": go_, "ll": ll(gi, go_)} for gi in grid for go_ in grid]
    bt = min(res["grid_total"], key=lambda x: x["ll"]); bs = min(res["grid_split"], key=lambda x: x["ll"])
    gk = np.concatenate([f[4] for f in folds.values()])
    ug, gi = np.unique(gk, return_inverse=True)
    rng = np.random.default_rng(20261027)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    cnt = np.bincount(gi).astype(float)
    base = ll(0, 0, per_row=True)
    res["vs_V5_park"] = {}
    for name, (x1, x2) in {"total": (bt["gamma"], bt["gamma"]), "split": (bs["g_if"], bs["g_of"])}.items():
        dlt = ll(x1, x2, per_row=True) - base
        b = (w @ np.bincount(gi, weights=dlt)) / (w @ cnt)
        res["vs_V5_park"][name] = {"g_if": x1, "g_of": x2, "mean": float(dlt.mean()),
                                   "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "grid_split"}, indent=1))


if __name__ == "__main__":
    main()
