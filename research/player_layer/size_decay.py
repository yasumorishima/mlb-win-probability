"""
Stage A with earlier seasons down-weighted (lam ** seasons ago), on top of
V5 + park; shrinkage re-chosen for each lam. 2022-2025, league tables from
the other four of 2021-2025, records causal. 2026 not read.

  python research/player_layer/size_decay.py --states S --hf HF --out <json>
"""

from __future__ import annotations

import argparse
import itertools
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
from rates_v2 import prior_class_counts_decay  # noqa: E402
from size_context import ABBR, PF_COLS  # noqa: E402

C = R.PA_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pa0 = R.load_pa(a.states, S.SEASONS)
    pa0 = pa0[pa0.cls.isin(C) & pa0.pa_complete].reset_index(drop=True)
    dc = S.defense_counts(pa0)
    pa0 = pa0.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    pa0["sk"] = S.state_key(pa0)
    pf = pd.read_parquet(a.hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    pf = pf.assign(season=pf.season + 1).set_index(["team", "season"])
    j = pf.reindex(pd.MultiIndex.from_arrays([pa0.home_team.map(ABBR), pa0.season]))
    PF = np.ones((len(pa0), len(C)))
    for c, col in PF_COLS.items():
        PF[:, C.index(c)] = j[col].to_numpy() / 100.0
    PF[((pa0.home_team.map(ABBR).isin(["OAK", "TB"])) & (pa0.season == 2025)).to_numpy()] = np.nan
    PF[np.isnan(PF)] = 1.0
    y = pa0.cls.map({c: i for i, c in enumerate(C)}).to_numpy()
    league = {s: S.league_tables(pa0[pa0.season != s]) for s in S.SCORE}
    g = pa0.game_pk.to_numpy()
    res = {"grid": []}
    per_row = {}
    for lam in (1.0, 0.8, 0.6, 0.4):
        pa = pa0.copy()
        for who in ("batter", "pitcher"):
            cnt = prior_class_counts_decay(pa, who, C, lam)
            pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
        folds = {}
        for s in S.SCORE:
            m = (pa.season == s).to_numpy(); d = pa[m]
            st, L = league[s]
            folds[s] = (st.reindex(d.sk).to_numpy() * PF[m] ** 0.5, d[[f"batter_{c}" for c in C]].to_numpy(),
                        d[[f"pitcher_{c}" for c in C]].to_numpy(), d[[f"def_{c}" for c in R.BIP]].to_numpy(), L, y[m])
        scale = [1.0, 0.7, 0.5] if lam < 1 else [1.0]
        for sc in scale:
            prm = {k: v * sc for k, v in Ly.PARAMS.items()}
            prm["kd"] = Ly.PARAMS["kd"]
            out = []
            for base, B, P, D, L, yy in folds.values():
                p = base * Ly.multipliers(B, P, D, L, prm=prm)
                p = p / p.sum(axis=1, keepdims=True)
                out.append(-np.log(p[np.arange(len(yy)), yy]))
            v = np.concatenate(out)
            res["grid"].append({"lam": lam, "k_scale": sc, "ll": float(v.mean())})
            per_row[(lam, sc)] = v
            print(lam, sc, round(float(v.mean()), 6), flush=True)
    best = min(res["grid"], key=lambda x: x["ll"])
    base = per_row[(1.0, 1.0)]
    dl = per_row[(best["lam"], best["k_scale"])] - base
    gg = np.concatenate([pa0[pa0.season == s].game_pk.to_numpy() for s in S.SCORE])
    ug, gi = np.unique(gg, return_inverse=True)
    rng = np.random.default_rng(20261028)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    bs = (w @ np.bincount(gi, weights=dl)) / (w @ np.bincount(gi).astype(float))
    res["best"] = best
    res["best_minus_no_decay"] = {"mean": float(dl.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("best", "best_minus_no_decay")}, indent=1))


if __name__ == "__main__":
    main()
