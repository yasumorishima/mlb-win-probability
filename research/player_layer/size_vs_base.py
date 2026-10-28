"""
size_vs_base.py: size_vs_pen.py (frozen in PREREG_PENW) with --base-tag. Compare a candidate variant of size_pen_x.py with a baseline (default PEN_RECAL) on 2022-2025 (sizing; 2026 not read).

Both are recalibrated the same way as PEN_RECAL (terms L, L r, L r^2, r, r^2), out of fold:
fitted on three of 2022-2025 and applied to the fourth.

  python research/player_layer/size_vs_pen.py --states S --base DIR_B --cand DIR_C --cand-tag penB_f4 --out O.json
      DIR_B: penB_2022..2025.parquet (PEN, the tested layer); DIR_C: <cand-tag>_2022..2025.parquet
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
from size_runs import game_boot  # noqa: E402
from size_stack import design, fit  # noqa: E402
from test_pen_2026 import TERMS, with_r  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]


def rows(states: Path, d: Path, tag: str) -> pd.DataFrame:
    cols = ["game_pk", "at_bat_index", "season", "inning", "half", "outs", "scheduled_innings", "home_won"]
    pa = pd.concat([pd.read_parquet(states / f"pa_{s}.parquet", columns=cols) for s in SEASONS], ignore_index=True)
    p = pd.concat([pd.read_parquet(d / f"{tag}_{s}.parquet") for s in SEASONS], ignore_index=True)
    m = pa.merge(p[["game_pk", "at_bat_index", "wp_new"]], on=["game_pk", "at_bat_index"], how="left")
    if m.wp_new.isna().any():
        sys.exit(f"{tag}: missing predictions")
    return with_r(m.rename(columns={"wp_new": "wp_game"}))


def oof(d: pd.DataFrame) -> np.ndarray:
    y = d.home_won.to_numpy(float)
    out = np.empty(len(d))
    for s in SEASONS:
        tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
        w = fit(design(d[tr], "wp_game", TERMS), y[tr])
        out[te] = 1 / (1 + np.exp(-(design(d[te], "wp_game", TERMS) @ w)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--cand-tag", required=True)
    ap.add_argument("--base-tag", default="penB", help="baseline files <base-tag>_<season> (penB = PEN_RECAL, penB_w60 = PENW_RECAL)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    b = rows(a.states, a.base, a.base_tag)
    c = rows(a.states, a.cand, a.cand_tag)
    assert (b[["game_pk", "at_bat_index"]].to_numpy() == c[["game_pk", "at_bat_index"]].to_numpy()).all()
    y = b.home_won.to_numpy(float)
    g = b.game_pk.to_numpy()
    season = b.season.to_numpy()
    rng = np.random.default_rng(20261031)
    res = {"n": int(len(b))}
    preds = {"PEN_RECAL": oof(b), "CAND_RECAL": oof(c), "PEN": b.wp_game.to_numpy(), "CAND": c.wp_game.to_numpy()}
    se = {k: (v - y) ** 2 for k, v in preds.items()}
    for k, v in preds.items():
        band = (v >= 0.3) & (v < 0.5)
        res[k] = {"brier": float(se[k].mean()), "band_0.3_0.5_bias_pt": float(100 * (v[band].mean() - y[band].mean())),
                  "by_season": {int(s): float(se[k][season == s].mean()) for s in SEASONS}}
    for name, (x, z) in {"CAND_RECAL_minus_PEN_RECAL": ("CAND_RECAL", "PEN_RECAL"), "CAND_minus_PEN": ("CAND", "PEN")}.items():
        dl = se[x] - se[z]
        res[name] = {"mean": float(dl.mean()), "ci95": game_boot(dl, g, rng),
                     "by_season": {int(s): float(dl[season == s].mean()) for s in SEASONS}}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if "minus" in k}, indent=1))
    for k in preds:
        print(k, round(res[k]["brier"], 6), "band", round(res[k]["band_0.3_0.5_bias_pt"], 2))


if __name__ == "__main__":
    main()
