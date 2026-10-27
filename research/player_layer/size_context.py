"""
Stage A, context terms on top of the best player variant (V5 multipliers
from layer.PARAMS): park and defense. 2022-2025, league tables from the
other four of 2021-2025, records causal. 2026 is not read.

  python research/player_layer/size_context.py --states <dir> --hf <dir with park_factors / oaa_team> --out <json>

  park : the home park's previous-season class factors (Savant 3-year
         pf_so, pf_bb, pf_1b, pf_2b, pf_3b, pf_hr; 100 = average), applied as
         (pf / 100) ** alpha to K, BB, 1B, 2B, 3B, HR.
  oaa  : the fielding team's previous-season team OAA, applied to the hit
         classes on balls in play as (1 - gamma * oaa / 100) (the scale is
         chosen by the grid on gamma).
The same season's factors are never used (Savant season tables include the
games being predicted).
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

ABBR = {133: "OAK", 134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL", 139: "TB", 140: "TEX",
        141: "TOR", 142: "MIN", 143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
        108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE", 115: "COL",
        116: "DET", 117: "HOU", 118: "KC", 119: "LAD", 120: "WSH", 121: "NYM"}   # Stats API ids (2025)
PF_COLS = {"K": "pf_so", "BB": "pf_bb", "1B": "pf_1b", "2B": "pf_2b", "3B": "pf_3b", "HR": "pf_hr"}
C = R.PA_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pa = R.load_pa(a.states, S.SEASONS)
    pa = pa[pa.cls.isin(C) & pa.pa_complete].reset_index(drop=True)
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C)
        pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
    dc = S.defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    pa["sk"] = S.state_key(pa)

    pf = pd.read_parquet(a.hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    pf = pf.assign(season=pf.season + 1).set_index(["team", "season"])          # previous season's factors
    pa["park"] = pa.home_team.map(ABBR)
    PF = np.ones((len(pa), len(C)))
    j = pf.reindex(pd.MultiIndex.from_arrays([pa.park, pa.season]))
    for c, col in PF_COLS.items():
        PF[:, C.index(c)] = (j[col].to_numpy() / 100.0)
    # new home parks in 2025 (Athletics: Sacramento; Rays: Steinbrenner Field): no previous-season factor
    PF[((pa.park.isin(["OAK", "TB"])) & (pa.season == 2025)).to_numpy()] = np.nan
    miss_pf = np.isnan(PF).any(axis=1)
    PF[np.isnan(PF)] = 1.0

    oaa = pd.read_parquet(a.hf / "oaa_team.parquet")
    oaa["team_abbrev"] = oaa.team_abbrev.replace({"ATH": "OAK"})
    oaa = oaa.assign(season=oaa.season + 1).set_index(["team_abbrev", "season"]).total_oaa
    OA = oaa.reindex(pd.MultiIndex.from_arrays([pa.fld_team.map(ABBR), pa.season])).to_numpy(float)
    miss_oaa = np.isnan(OA)
    OA[miss_oaa] = 0.0

    y = pa.cls.map({c: i for i, c in enumerate(C)}).to_numpy()
    hit = [C.index(c) for c in ("1B", "2B", "3B")]
    folds = {}
    for s in S.SCORE:
        st, L = S.league_tables(pa[pa.season != s])
        m = (pa.season == s).to_numpy()
        d = pa[m]
        M = Ly.multipliers(d[[f"batter_{c}" for c in C]].to_numpy(), d[[f"pitcher_{c}" for c in C]].to_numpy(),
                           d[[f"def_{c}" for c in R.BIP]].to_numpy(), L)
        folds[s] = (st.reindex(d.sk).to_numpy() * M, PF[m], OA[m], y[m], d.game_pk.to_numpy())

    def ll(alpha, gamma, per_row=False):
        out = []
        for base, pfm, oa, yy, _ in folds.values():
            p = base * pfm ** alpha
            p[:, hit] *= (1 - gamma * oa / 100.0)[:, None]
            p = p / p.sum(axis=1, keepdims=True)
            out.append(-np.log(p[np.arange(len(yy)), yy]))
        v = np.concatenate(out)
        return v if per_row else float(v.mean())

    res = {"missing_park_rows": int(miss_pf.sum()), "missing_oaa_rows": int(miss_oaa.sum()), "grid": []}
    for alpha in (0.0, 0.5, 1.0):
        for gamma in (0.0, 0.1, 0.2, 0.4):
            res["grid"].append({"alpha": alpha, "gamma": gamma, "ll": ll(alpha, gamma)})
    best = min(res["grid"], key=lambda x: x["ll"])
    g = np.concatenate([f[4] for f in folds.values()])
    base = ll(0.0, 0.0, per_row=True)
    rng = np.random.default_rng(20261027)
    ug, gi = np.unique(g, return_inverse=True)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    cnt = np.bincount(gi).astype(float)
    res["vs_V5"] = {}
    for name, (al, ga) in {"park": (best["alpha"], 0.0), "oaa": (0.0, best["gamma"]),
                           "both": (best["alpha"], best["gamma"])}.items():
        dlt = ll(al, ga, per_row=True) - base
        bs = (w @ np.bincount(gi, weights=dlt)) / (w @ cnt)
        res["vs_V5"][name] = {"alpha": al, "gamma": ga, "mean": float(dlt.mean()),
                              "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
