"""
Sizing of stage A (PA outcome probabilities) on 2022-2025, each season
scored with league rates fitted on the other four of 2021-2025.
Player and team records use only dates before the PA (causal), over all
of 2021-2025 that precede it. 2026 is not read.

  python research/player_layer/size_pa.py --states <dir> --out <json>

Variants (p proportional to L(c | half, base-out state) times multipliers):
  V0  league only (the engine's assumption)
  V1  x batter
  V2  x batter x pitcher (all classes, log5)
  V3  x batter; pitcher only on K / BB / HR / ball-in-play total;
      fielding team on how balls in play end (1B / 2B / 3B / OUT)
  V4  V3, plus the pitcher's own ball-in-play split (shrunk with kpb)
Metric: mean multinomial log loss per PA (lower is better), and the
paired difference vs V0 with a game-level bootstrap interval.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pa_rates as R  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SCORE = [2022, 2023, 2024, 2025]
C = R.PA_CLASSES
IB = [C.index(c) for c in R.BIP]
NB = [C.index(c) for c in R.NONBIP]


def state_key(df):
    return (df.half + "|" + df.outs.astype(str) + df.on_1b.notna().astype(int).astype(str)
            + df.on_2b.notna().astype(int).astype(str) + df.on_3b.notna().astype(int).astype(str))


def league_tables(train: pd.DataFrame):
    t = train[train.cls.isin(C) & train.pa_complete]
    st = pd.crosstab(state_key(t), t.cls).reindex(columns=C, fill_value=0) + 0.5
    st = st.div(st.sum(axis=1), axis=0)
    overall = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    return st, overall


def defense_counts(pa: pd.DataFrame) -> pd.DataFrame:
    """Fielding team's ball-in-play outcomes earlier this season."""
    rows = pa[pa.cls.isin(R.BIP) & pa.pa_complete]
    d = pd.crosstab([rows.fld_team, rows.season, rows.date], rows.cls).reindex(columns=R.BIP, fill_value=0).sort_index()
    thru = d.groupby(level=[0, 1]).cumsum().reset_index()
    thru["date"] = thru["date"] + pd.Timedelta(days=1)
    q = pa[["fld_team", "season", "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(q, thru.sort_values("date"), on="date", by=["fld_team", "season"], direction="backward")
    m[R.BIP] = m[R.BIP].fillna(0).astype(float)
    return m


def probs(variant, Ls, L, B, P, D, kb, kp, kd, kpb):
    """Ls: league P(c|state) per row; L: overall league rates; B/P: raw
    class counts of batter / pitcher; D: fielding team's BIP counts."""
    if variant == "V0":
        p = Ls.copy()
    else:
        mb = R.shrunk(B, L, kb) / L
        p = Ls * mb
        if variant == "V2":
            p = p * (R.shrunk(P, L, kp) / L)
        elif variant in ("V3", "V4"):
            # pitcher on K / BB / HR / BIP-total
            Lc = np.concatenate([L[NB], [L[IB].sum()]])
            Pc = np.column_stack([P[:, NB], P[:, IB].sum(axis=1)])
            mp = R.shrunk(Pc, Lc, kp) / Lc
            p[:, NB] *= mp[:, :3]
            p[:, IB] *= mp[:, 3:4]
            Lb = L[IB] / L[IB].sum()
            p[:, IB] *= R.shrunk(D, Lb, kd) / Lb
            if variant == "V4":
                p[:, IB] *= R.shrunk(P[:, IB], Lb, kpb) / Lb
    return p / p.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pa = R.load_pa(a.states, SEASONS)
    pa = pa[pa.cls.isin(C) & pa.pa_complete].reset_index(drop=True)
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C)
        pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
    dc = defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    pa["sk"] = state_key(pa)
    y = pa.cls.map({c: i for i, c in enumerate(C)}).to_numpy()

    grid = {"kb": [100, 200, 400, 800], "kp": [100, 200, 400, 800], "kd": [500, 1000, 2000, 4000],
            "kpb": [500, 1000, 2000, 4000]}
    out = {"n_scored": {}, "results": []}
    folds = {}
    for s in SCORE:
        st, L = league_tables(pa[pa.season.isin([x for x in SEASONS if x != s])])
        m = (pa.season == s).to_numpy()
        d = pa[m]
        folds[s] = dict(Ls=st.reindex(d.sk).to_numpy(), L=L,
                        B=d[[f"batter_{c}" for c in C]].to_numpy(), P=d[[f"pitcher_{c}" for c in C]].to_numpy(),
                        D=d[[f"def_{c}" for c in R.BIP]].to_numpy(), y=y[m], g=d.game_pk.to_numpy())
        out["n_scored"][s] = int(m.sum())

    def total_ll(variant, kb=200, kp=200, kd=1000, kpb=1000, per_row=False):
        lls = []
        for s, f in folds.items():
            p = probs(variant, f["Ls"], f["L"], f["B"], f["P"], f["D"], kb, kp, kd, kpb)
            lls.append(-np.log(np.clip(p[np.arange(len(f["y"])), f["y"]], 1e-12, 1)))
        v = np.concatenate(lls)
        return v if per_row else float(v.mean())

    base = total_ll("V0")
    out["results"].append({"variant": "V0", "ll": base})
    best = {}
    for kb in grid["kb"]:
        r = total_ll("V1", kb=kb); out["results"].append({"variant": "V1", "kb": kb, "ll": r})
    best["kb"] = min((x for x in out["results"] if x["variant"] == "V1"), key=lambda x: x["ll"])["kb"]
    for kp in grid["kp"]:
        r = total_ll("V2", kb=best["kb"], kp=kp); out["results"].append({"variant": "V2", "kb": best["kb"], "kp": kp, "ll": r})
    for kp, kd in itertools.product(grid["kp"], grid["kd"]):
        r = total_ll("V3", kb=best["kb"], kp=kp, kd=kd)
        out["results"].append({"variant": "V3", "kb": best["kb"], "kp": kp, "kd": kd, "ll": r})
    b3 = min((x for x in out["results"] if x["variant"] == "V3"), key=lambda x: x["ll"])
    for kpb in grid["kpb"]:
        r = total_ll("V4", kb=b3["kb"], kp=b3["kp"], kd=b3["kd"], kpb=kpb)
        out["results"].append({"variant": "V4", "kb": b3["kb"], "kp": b3["kp"], "kd": b3["kd"], "kpb": kpb, "ll": r})

    # paired game-bootstrap of the best of each variant vs V0
    g = np.concatenate([f["g"] for f in folds.values()])
    l0 = total_ll("V0", per_row=True)
    rng = np.random.default_rng(20261027)
    ug, gi = np.unique(g, return_inverse=True)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    cnt = np.bincount(gi).astype(float)
    out["vs_V0"] = {}
    for v in ("V1", "V2", "V3", "V4"):
        bx = min((x for x in out["results"] if x["variant"] == v), key=lambda x: x["ll"])
        kw = {k: bx[k] for k in ("kb", "kp", "kd", "kpb") if k in bx}
        dlt = total_ll(v, per_row=True, **kw) - l0
        bs = (w @ np.bincount(gi, weights=dlt)) / (w @ cnt)
        out["vs_V0"][v] = {"params": kw, "mean": float(dlt.mean()),
                           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["vs_V0"], indent=1), "V0", base)


if __name__ == "__main__":
    main()
