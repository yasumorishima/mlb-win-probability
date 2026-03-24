"""
Compute conformal prediction quantiles for Statcast WP model.

Split conformal: calibration on holdout data not seen during training.
Produces prediction intervals with valid marginal coverage guarantee.

Usage:
  python scripts/compute_conformal.py --data-dir data/ --output-dir data/
  python scripts/compute_conformal.py --data-dir data/ --output-dir data/ --cal-year 2024
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_conformal_quantiles(
    predictions: np.ndarray,
    actuals: np.ndarray,
    alphas: list[float] = [0.01, 0.05, 0.10, 0.20],
) -> dict:
    """Compute conformal quantiles from calibration residuals.

    Args:
        predictions: Model predicted probabilities (N,)
        actuals: Binary outcomes 0/1 (N,)
        alphas: Miscoverage levels (e.g., 0.10 → 90% interval)

    Returns:
        Dict with quantile values and diagnostics.
    """
    n = len(predictions)
    scores = np.abs(predictions - actuals)

    results = {"n_calibration": n, "quantiles": {}}
    for alpha in alphas:
        level = 1 - alpha
        # Conformal quantile: ceil((n+1)*(1-alpha)) / n
        idx = int(math.ceil((n + 1) * level)) - 1
        idx = min(idx, n - 1)
        q = float(np.sort(scores)[idx])

        # Empirical coverage check
        intervals_lo = np.clip(predictions - q, 0, 1)
        intervals_hi = np.clip(predictions + q, 0, 1)
        covered = ((actuals >= intervals_lo) & (actuals <= intervals_hi)).mean()

        results["quantiles"][f"{level:.0%}"] = {
            "alpha": alpha,
            "q": round(q, 6),
            "empirical_coverage": round(float(covered), 4),
            "mean_interval_width": round(float(2 * q), 4),
        }
        print(f"  {level:.0%} interval: q={q:.4f}, "
              f"width={2*q:.4f}, coverage={covered:.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute conformal prediction quantiles")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--cal-year", type=int, default=2024,
                        help="Calibration year (must be holdout from training)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Statcast model
    from win_probability_statcast import WPEngineStatcast
    engine = WPEngineStatcast()
    if not engine.is_loaded:
        print("ERROR: Statcast model not found")
        sys.exit(1)

    # Load calibration data
    from scripts.ensemble_wp import load_play_states
    cal_path = data_dir / f"play_states_{args.cal_year}.csv"
    if not cal_path.exists():
        print(f"ERROR: {cal_path} not found")
        sys.exit(1)

    states = load_play_states(cal_path)
    print(f"Calibration data: {len(states):,} plays ({args.cal_year})")

    # Batch predict
    features = []
    for s in states:
        gs = {
            "inning": s["inning"],
            "top_bottom": "bottom" if s["half_inning"] == "bottom" else "top",
            "outs": s["outs"],
            "runners": s["runners"],
            "score_diff": s["score_diff"],
            "balls": 0, "strikes": 0,
        }
        features.append(engine._build_features(gs, None, None))

    X = np.array(features, dtype=np.float32)
    predictions = np.clip(engine._model.predict(X), 0.001, 0.999)
    actuals = np.array([s["home_won"] for s in states], dtype=float)

    print(f"Predictions: mean={predictions.mean():.4f}, "
          f"std={predictions.std():.4f}")

    # Compute conformal quantiles
    print(f"\nConformal quantiles:")
    results = compute_conformal_quantiles(predictions, actuals)
    results["calibration_year"] = args.cal_year
    results["model"] = "statcast_lgbm"

    # Brier score for reference
    brier = float(np.mean((predictions - actuals) ** 2))
    results["brier_on_calibration"] = round(brier, 6)
    print(f"\nBrier on calibration set: {brier:.6f}")

    # Save
    out_path = output_dir / "conformal_quantiles.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
