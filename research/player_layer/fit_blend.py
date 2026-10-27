"""
Fit the BLEND model on the 2022-2025 out-of-fold rows and write its
coefficients (no 2026 data is read):

  BLEND = sigmoid( w * logit(P_RECAL) + (1 - w) * logit(E_TEAM) )
  P_RECAL = sigmoid(X_recal(player-layer WP) . a)       terms L, L r, L r^2, r, r^2
  E_TEAM  = sigmoid(X_team(engine WP) . b)              + d_prev r, d_prev r^2, d_cur r, d_cur r^2
w on a 0.05 grid by Brier on all four seasons. Inputs: size_game.py's
per-PA rows (player layer v1 `wp_game`, engine `wp_league`, both out of
fold) and research/wp_layer/layer_features.py team strength.

  python research/player_layer/fit_blend.py --states S --game <size_game out dir> --out research/player_layer/published/blend_coef.json
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
sys.path.insert(0, str(HERE.parents[1] / "research" / "wp_layer"))
from size_stack import ARMS, SEASONS, design, fit, logit  # noqa: E402
from layer_features import game_table, load_pa, team_strength  # noqa: E402


def rows_with_features(states: Path, rows: pd.DataFrame, seasons) -> pd.DataFrame:
    pa = load_pa(states, sorted(set(seasons) | {min(seasons) - 1}))
    ts = team_strength(game_table(pa))
    pa = pa[pa.season.isin(seasons)][["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings",
                                       "home_score", "away_score"]]
    d = rows.merge(pa, on=["game_pk", "at_bat_index"], how="left", suffixes=("", "_pa")).join(ts, on="game_pk")
    total = 6 * d.scheduled_innings
    done = ((d.inning - 1) * 2 + (d.half == "bottom")) * 3 + d.outs
    d["r"] = np.maximum(1 - done / total, 3 / total)
    d["diff"] = np.clip(d.home_score - d.away_score, -10, 10)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--game", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = pd.concat([pd.read_parquet(a.game / f"game_{s}.parquet") for s in SEASONS], ignore_index=True)
    d = rows_with_features(a.states, rows, SEASONS)
    assert d[["d_prev", "d_cur", "r"]].notna().all().all() and len(d) == 733260
    y = d.home_won.to_numpy(float)
    out = {"fitted_on": SEASONS, "n": int(len(d)), "player_input": "size_game.py wp_game (v1, out of fold)",
           "engine_input": "size_game.py wp_league (out of fold)"}
    preds = {}
    for name in ("P_RECAL", "E_TEAM"):
        base, terms = ARMS[name]
        w = fit(design(d, base, terms), y)
        out[name] = {"base": base, "terms": terms, "coef": [float(x) for x in w]}
        preds[name] = 1 / (1 + np.exp(-(design(d, base, terms) @ w)))
    lp, le = logit(preds["P_RECAL"]), logit(preds["E_TEAM"])
    grid = np.round(np.linspace(0, 1, 21), 2)
    sc = [float(np.mean((1 / (1 + np.exp(-(w * lp + (1 - w) * le))) - y) ** 2)) for w in grid]
    out["weight_player"] = float(grid[int(np.argmin(sc))])
    out["in_sample_brier"] = {"blend": min(sc), "P_RECAL": float(np.mean((preds["P_RECAL"] - y) ** 2)),
                              "E_TEAM": float(np.mean((preds["E_TEAM"] - y) ** 2))}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
