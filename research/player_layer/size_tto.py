"""
Times through the order (sizing on 2022-2025, each season scored with factors and
league tables from the other four of 2021-2025; 2026 not read).

Base: the player layer's PA outcome probabilities (league rate for the half and
base-out state x layer.multipliers with the frozen PARAMS: batter, pitcher, defense).
Candidate: base x f[group, class], renormalised, where group = the starter's time
through the order (1, 2, 3, 4+, from the batters he has faced in this game) or
"relief"; f = observed / expected class counts of the base in the training seasons.

  python research/player_layer/size_tto.py --states S --out O.json
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
from size_pa import defense_counts, league_tables, state_key  # noqa: E402
from size_runs import game_boot  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
GROUPS = ["tto1", "tto2", "tto3", "tto4+", "relief"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pa = R.load_pa(a.states, SEASONS).sort_values(["game_pk", "at_bat_index"]).reset_index(drop=True)
    done = (pa.cls.isin(C) & pa.pa_complete).astype(int)
    pa["bf_before"] = done.groupby([pa.game_pk, pa.pitcher]).cumsum() - done
    starter = pa.groupby(["game_pk", "fld_team"]).pitcher.transform("first")
    tto = np.minimum(pa.bf_before // 9 + 1, 4)
    pa["grp"] = np.where(pa.pitcher == starter, tto.map({1: 0, 2: 1, 3: 2, 4: 3}), 4)
    pa = pa[pa.cls.isin(C) & pa.pa_complete].reset_index(drop=True)
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C)
        pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
    dc = defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    pa["sk"] = state_key(pa)
    y = pa.cls.map({c: i for i, c in enumerate(C)}).to_numpy()
    Y = np.eye(len(C))[y]
    grp = pa.grp.to_numpy()
    season = pa.season.to_numpy()
    B = pa[[f"batter_{c}" for c in C]].to_numpy()
    P = pa[[f"pitcher_{c}" for c in C]].to_numpy()
    D = pa[[f"def_{c}" for c in R.BIP]].to_numpy()

    ll0, ll1, gs, ss, factors = [], [], [], [], {}
    for s in SCORE:
        train = season != s
        st, L = league_tables(pa[train & (season != 2026)])
        p0 = st.reindex(pa.sk).to_numpy() * Ly.multipliers(B, P, D, L)
        p0 /= p0.sum(axis=1, keepdims=True)
        f = np.ones((len(GROUPS), len(C)))
        for k in range(len(GROUPS)):
            m = train & (grp == k)
            f[k] = Y[m].sum(axis=0) / p0[m].sum(axis=0)
        factors[s] = f.round(4).tolist()
        p1 = p0 * f[grp]
        p1 /= p1.sum(axis=1, keepdims=True)
        te = season == s
        ll0.append(-np.log(p0[te, y[te]])); ll1.append(-np.log(p1[te, y[te]]))
        gs.append(pa.game_pk.to_numpy()[te]); ss.append(np.full(te.sum(), s))
    l0, l1, g, sv = map(np.concatenate, (ll0, ll1, gs, ss))
    gv = np.concatenate([grp[season == s] for s in SCORE])
    d = l1 - l0
    rng = np.random.default_rng(20261031)
    res = {"n": int(len(d)), "classes": C, "groups": GROUPS, "base_ll": float(l0.mean()), "tto_ll": float(l1.mean()),
           "tto_minus_base": {"mean": float(d.mean()), "ci95": game_boot(d, g, rng),
                              "by_season": {int(s): float(d[sv == s].mean()) for s in SCORE},
                              "by_group": {GROUPS[k]: {"n": int((gv == k).sum()), "mean": float(d[gv == k].mean())} for k in range(len(GROUPS))}},
           "factors_by_fold": factors}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "factors_by_fold"}, indent=1))
    print("factors (fold 2025):", dict(zip(GROUPS, factors[2025])))


if __name__ == "__main__":
    main()
