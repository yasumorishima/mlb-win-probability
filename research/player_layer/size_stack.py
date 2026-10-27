"""
Final-stage calibration on top of the player layer, sized on 2022-2025.

The player layer's per-PA WP for 2022-2025 (size_game.py, `wp_game`) is
out-of-fold: each season was predicted with tables fitted on the other
four. A logistic layer on top can therefore learn the real out-of-sample
error (unlike the team layer's RECAL, fitted on in-sample engine WP).
Each layer is fitted on three of 2022-2025 and scored on the fourth.

  python research/player_layer/size_stack.py --states S --game <size_game out dir> --out <json>

Arms (logistic regression, unpenalised, terms plus intercept):
  P        the player layer as it is
  P_RECAL  L, L r, L r^2, r, r^2                 (L = logit of the player WP)
  P_TEAM   P_RECAL + d_prev r, d_prev r^2, d_cur r, d_cur r^2
  P_GAME   P_TEAM + diff r, diff r^2
  E_TEAM   the same TEAM terms on the engine's out-of-fold WP (reference)
d_prev / d_cur / r as in research/wp_layer/layer_features.py (causal).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research" / "wp_layer"))
from layer_features import game_table, load_pa, team_strength  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
CLIP = 1e-4


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def design(d: pd.DataFrame, base: str, terms: list[str]) -> np.ndarray:
    L = logit(d[base].to_numpy())
    r = d.r.to_numpy()
    cols = {"L": L, "Lr": L * r, "Lr2": L * r * r, "r": r, "r2": r * r,
            "dprev_r": d.d_prev.to_numpy() * r, "dprev_r2": d.d_prev.to_numpy() * r * r,
            "dcur_r": d.d_cur.to_numpy() * r, "dcur_r2": d.d_cur.to_numpy() * r * r,
            "diff_r": d["diff"].to_numpy() * r, "diff_r2": d["diff"].to_numpy() * r * r}
    return np.column_stack([np.ones(len(d))] + [cols[t] for t in terms])


def fit(X, y):
    def f(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        return np.sum(np.logaddexp(0, z) - y * z), X.T @ (p - y)
    w0 = np.zeros(X.shape[1]); w0[1] = 1.0
    res = minimize(f, w0, jac=True, method="L-BFGS-B")
    return res.x


ARMS = {
    "P_RECAL": ("wp_game", ["L", "Lr", "Lr2", "r", "r2"]),
    "P_TEAM": ("wp_game", ["L", "Lr", "Lr2", "r", "r2", "dprev_r", "dprev_r2", "dcur_r", "dcur_r2"]),
    "P_GAME": ("wp_game", ["L", "Lr", "Lr2", "r", "r2", "dprev_r", "dprev_r2", "dcur_r", "dcur_r2", "diff_r", "diff_r2"]),
    "E_TEAM": ("wp_league", ["L", "Lr", "Lr2", "r", "r2", "dprev_r", "dprev_r2", "dcur_r", "dcur_r2"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--game", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = pd.concat([pd.read_parquet(a.game / f"game_{s}.parquet") for s in SEASONS], ignore_index=True)
    pa = load_pa(a.states, [2021] + SEASONS)
    games = game_table(pa)
    ts = team_strength(games)
    pa = pa[pa.season.isin(SEASONS)][["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings",
                                       "home_score", "away_score"]]
    d = rows.merge(pa, on=["game_pk", "at_bat_index"], how="left").join(ts, on="game_pk")
    total = 6 * d.scheduled_innings
    done = ((d.inning - 1) * 2 + (d.half == "bottom")) * 3 + d.outs
    d["r"] = np.maximum(1 - done / total, 3 / total)
    d["diff"] = np.clip(d.home_score - d.away_score, -10, 10)
    assert d[["d_prev", "d_cur", "r"]].notna().all().all()
    y = d.home_won.to_numpy(float)
    pred = {"P": d.wp_game.to_numpy(), "E": d.wp_league.to_numpy()}
    for arm, (base, terms) in ARMS.items():
        p = np.empty(len(d))
        for s in SEASONS:
            tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
            w = fit(design(d[tr], base, terms), y[tr])
            p[te] = 1 / (1 + np.exp(-(design(d[te], base, terms) @ w)))
        pred[arm] = p
    # blends of the two out-of-fold layers: average in logit space, weight chosen per fold on the other seasons
    lp, le = logit(pred["P_RECAL"]), logit(pred["E_TEAM"])
    for name in ("BLEND",):
        p = np.empty(len(d)); ws = {}
        for s in SEASONS:
            tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
            grid = np.linspace(0, 1, 21)
            sc = [np.mean((1 / (1 + np.exp(-(w * lp[tr] + (1 - w) * le[tr]))) - y[tr]) ** 2) for w in grid]
            w = grid[int(np.argmin(sc))]; ws[int(s)] = float(w)
            p[te] = 1 / (1 + np.exp(-(w * lp[te] + (1 - w) * le[te])))
        pred[name] = p
    g = d.game_pk.to_numpy()
    ug, gi = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(20261028)
    wts = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    cnt = wts @ np.bincount(gi).astype(float)
    se = {k: (v - y) ** 2 for k, v in pred.items()}
    res = {"n": int(len(d)), "blend_weight_on_player_by_fold": ws}
    for k, v in pred.items():
        band = (v >= 0.3) & (v < 0.5)
        lo = v < 0.3
        res[k] = {"brier": float(se[k].mean()), "band_0.3_0.5_bias_pt": float(100 * (v[band].mean() - y[band].mean())),
                  "below_0.3_bias_pt": float(100 * (v[lo].mean() - y[lo].mean())),
                  "by_season": {int(s): float(se[k][d.season.to_numpy() == s].mean()) for s in SEASONS}}
        if k != "P":
            dl = se[k] - se["P"]
            bs = (wts @ np.bincount(gi, weights=dl)) / cnt
            res[k]["minus_P"] = {"mean": float(dl.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    a.out.write_text(json.dumps(res, indent=1))
    print("blend weights on the player layer:", ws)
    for k, v in res.items():
        if isinstance(v, dict) and "brier" in v:
            print(k, round(v["brier"], 6), "band", round(v["band_0.3_0.5_bias_pt"], 2), "low", round(v["below_0.3_bias_pt"], 2),
                  v.get("minus_P", ""))


if __name__ == "__main__":
    main()
