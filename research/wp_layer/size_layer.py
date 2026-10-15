"""
Sizing on the training seasons only (2022-2025, leave one season out).
2026 is not loaded. Prints Brier / log loss / band gaps per arm and fold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_features import build  # noqa: E402

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
from wp_engine import Engine  # noqa: E402

STATES = Path.home() / "claude-scratch/mlbwp/data/states"
SEASONS = [2022, 2023, 2024, 2025]


def design(d: pd.DataFrame, mu: float) -> dict:
    p = np.clip(d.p_eng.to_numpy(), 1e-4, 1 - 1e-4)
    L = np.log(p / (1 - p))
    r = d.r.to_numpy()
    re = d.runs_so_far.to_numpy() - mu * d.halves_done.to_numpy()
    c = {
        "L": L, "Lr": L * r, "Lr2": L * r * r, "r": r, "r2": r * r,
        "dprev_r": d.d_prev.to_numpy() * r, "dprev_r2": d.d_prev.to_numpy() * r * r,
        "dcur_r": d.d_cur.to_numpy() * r, "dcur_r2": d.d_cur.to_numpy() * r * r,
        "sp_r": d.sp.to_numpy() * r, "sp_r2": d.sp.to_numpy() * r * r,
        "diff_r": d["diff"].to_numpy() * r, "re": re, "L_re": L * re,
    }
    return c


BASE = ["L", "Lr", "Lr2", "r", "r2"]
ARMS = {
    "RECAL": BASE,
    "TEAM": BASE + ["dprev_r", "dprev_r2", "dcur_r", "dcur_r2"],
    "SP": BASE + ["sp_r", "sp_r2"],
    "INGAME": BASE + ["diff_r", "re", "L_re"],
    "FULL": BASE + ["dprev_r", "dprev_r2", "dcur_r", "dcur_r2", "sp_r", "sp_r2",
                    "diff_r", "re", "L_re"],
}


def fit(c: dict, cols, y):
    X = np.column_stack([c[k] for k in cols])
    m, s = X.mean(0), X.std(0)
    lr = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8)
    lr.fit((X - m) / s, y)
    return lambda cc: lr.predict_proba((np.column_stack([cc[k] for k in cols]) - m) / s)[:, 1]


def metrics(p, y, band_by=None):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {"brier": float(np.mean((p - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    for name, ref in (("own", p), ("eng", band_by)):
        if ref is None:
            continue
        m = (ref >= 0.3) & (ref < 0.5)
        out[f"gap35_{name}"] = float(p[m].mean() - y[m].mean())
    return out


def main():
    t0 = time.time()
    eng = Engine.load(APP / "engine_tables.json")
    d = build(STATES, SEASONS, eng)
    print("rows", len(d), "built in", round(time.time() - t0), "s", flush=True)
    print("has_prev share", float(d.has_prev.mean()), "sp nan", int(d.sp.isna().sum()),
          "dcur nan", int(d.d_cur.isna().sum()), flush=True)
    print(d[["d_prev", "d_cur", "sp", "r", "diff", "runs_so_far"]].describe().round(3).to_string())
    res = {}
    for hold in SEASONS:
        tr, te = d[d.season != hold], d[d.season == hold]
        halves = tr.groupby(["game_pk", "inning", "half"]).ngroups
        mu = float(tr.runs_on_play.sum() / halves)
        ctr, cte = design(tr, mu), design(te, mu)
        ytr, yte = tr.home_won.to_numpy(float), te.home_won.to_numpy(float)
        pe = te.p_eng.to_numpy()
        res[hold] = {"ENGINE": metrics(pe, yte, pe), "mu": mu}
        for arm, cols in ARMS.items():
            res[hold][arm] = metrics(fit(ctr, cols, ytr)(cte), yte, pe)
        print(hold, json.dumps({k: (v if isinstance(v, float) else {m: round(x, 5) for m, x in v.items()})
                                for k, v in res[hold].items()}), flush=True)
    for arm in ["ENGINE"] + list(ARMS):
        b = np.mean([res[h][arm]["brier"] for h in SEASONS])
        g = np.mean([res[h][arm]["gap35_eng"] for h in SEASONS])
        go = np.mean([res[h][arm]["gap35_own"] for h in SEASONS])
        print(f"{arm:7s} brier {b:.5f}  gap35 eng-band {g:+.4f} own-band {go:+.4f}")
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
