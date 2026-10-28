"""
Per-plate-appearance PEN_RECAL home win probability for the look-back view
(registered consequence of PREREG_PEN.md / RESULTS_PEN_2026.md).

  python research/player_layer/build_pen_wp.py --states S --pen P
      P = dir with size_pen.py variant B outputs penB_2022..2025.parquet (sizing)
  writes published/pen_wp_<season>.parquet (game_pk, at_bat_index, wp) for 2022-2026

2022-2025: the sizing predictions (tables fitted on the other seasons of
2021-2025, records from earlier dates); 2026: the frozen test predictions
(published/penB_2026_part*.parquet). All recalibrated with
published/pen_recal_coef.json (fitted on 2022-2025). WP before the plate
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
from test_pen_2026 import PEN26_FILES, apply, with_r  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--pen", type=Path, required=True)
    a = ap.parse_args()
    coef = json.loads((HERE / "published/pen_recal_coef.json").read_text())
    cols = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings"]
    for s in (2022, 2023, 2024, 2025, 2026):
        if s == 2026:
            pen = pd.concat([pd.read_parquet(f) for f in PEN26_FILES], ignore_index=True)
        else:
            pen = pd.read_parquet(a.pen / f"penB_{s}.parquet")
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
