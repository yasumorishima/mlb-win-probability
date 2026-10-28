"""
Per-plate-appearance home win probability for the look-back view: PENR_RECAL
(60-day usage, five lead states, KAPPA 0.25; registered consequence of PREREG_PENR.md /
RESULTS_PENR_2026.md; before that PENW_RECAL, PREREG_PENW.md, and PEN_RECAL, PREREG_PEN.md).

  python research/player_layer/build_pen_wp.py --states S --pen P
      P = dir with size_pen_x2.py --window 60 --lead5 --kappa 0.25 outputs penB_w60_l5_k0_2022..2025.parquet
  (--tag penB_w60 --coef published/penw_recal_coef.json rebuilds the PENW_RECAL files,
   --tag penB --coef published/pen_recal_coef.json the PEN_RECAL ones)
  writes published/pen_wp_<season>.parquet (game_pk, at_bat_index, wp) for 2022-2026

2022-2025: the sizing predictions (tables fitted on the other seasons of
2021-2025, records from earlier dates); 2026: the frozen test predictions
(published/<tag>_2026_part*.parquet). All recalibrated with the
coefficients fitted on 2022-2025. WP before the plate
appearance, home team, rounded to 1e-4.
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
from test_pen_2026 import apply, with_r  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--pen", type=Path, required=True)
    ap.add_argument("--tag", default="penB_w60_l5_k0")
    ap.add_argument("--coef", type=Path, default=HERE / "published/penr_recal_coef.json")
    a = ap.parse_args()
    coef = json.loads(a.coef.read_text())
    cols = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings"]
    for s in (2022, 2023, 2024, 2025, 2026):
        if s == 2026:
            pen = pd.concat([pd.read_parquet(HERE / f"published/{a.tag}_2026_part{k}.parquet") for k in range(4)],
                            ignore_index=True)
        else:
            pen = pd.read_parquet(a.pen / f"{a.tag}_{s}.parquet")
        pa = pd.read_parquet(a.states / f"pa_{s}.parquet", columns=cols)
        d = pa.merge(pen[["game_pk", "at_bat_index", "wp_new"]], on=["game_pk", "at_bat_index"], how="left")
        if len(d) != len(pa) or d.wp_new.isna().any():
            sys.exit(f"{s}: predictions do not cover the plate appearances")
        d = with_r(d.rename(columns={"wp_new": "wp_game"}))
        out = pd.DataFrame({"game_pk": d.game_pk.astype("int32"), "at_bat_index": d.at_bat_index.astype("int16"),
                            "wp": np.round(apply(coef, d), 4).astype("float32")})
        if not np.isfinite(out.wp).all():
            sys.exit("non-finite")
        p = HERE / f"published/pen_wp_{s}.parquet"
        out.sort_values(["game_pk", "at_bat_index"]).to_parquet(p, index=False, compression="zstd")
        print(s, len(out), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
