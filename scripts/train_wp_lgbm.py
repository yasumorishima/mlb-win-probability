"""
Train LightGBM model for WP prediction (Model C in the A+B+C approach).

Time-based holdout split:
  Train: all years except most recent
  Test:  most recent year

Features: inning, is_bottom, outs, r1, r2, r3, score_diff + derived
Target:   home_won (binary)

Usage:
  python scripts/train_wp_lgbm.py [--data-dir data/] [--output-dir data/]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "inning", "is_bottom", "outs", "r1", "r2", "r3", "score_diff",
    "inn_x_bottom", "abs_lead", "total_runners", "tied_innings_left",
]


def load_play_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, labels, and years from all play_states CSVs."""
    features = []
    labels = []
    years = []

    for csv_path in sorted(data_dir.glob("play_states_*.csv")):
        year = int(csv_path.stem.split("_")[-1])
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    inning = int(row["inning"])
                    is_bottom = 1 if row["half_inning"] == "bottom" else 0
                    outs = int(row["outs"])
                    r1 = int(row["runner_1b"])
                    r2 = int(row["runner_2b"])
                    r3 = int(row["runner_3b"])
                    score_diff = int(row["score_diff"])
                    home_won = int(row["home_won"])

                    features.append([
                        inning,
                        is_bottom,
                        outs,
                        r1, r2, r3,
                        score_diff,
                        # Derived features
                        inning * is_bottom,          # inning-half interaction
                        abs(score_diff),             # absolute lead size
                        r1 + r2 + r3,                # total runners on base
                        max(9 - inning, 0) * (1 if score_diff == 0 else 0),
                    ])
                    labels.append(home_won)
                    years.append(year)
                except (ValueError, KeyError):
                    continue

    return np.array(features), np.array(labels), np.array(years)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train LightGBM WP model (Model C)")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory with play_states CSVs")
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for model + metrics")
    parser.add_argument("--test-year", type=int, default=None,
                        help="Holdout year (default: most recent)")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. pip install lightgbm")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading play-by-play data...")
    X, y, years = load_play_data(data_dir)
    if len(X) == 0:
        print("ERROR: No data found")
        sys.exit(1)
    print(f"  Total: {len(X):,} plays, years {years.min()}-{years.max()}")

    # Split
    test_year = args.test_year or int(years.max())
    train_mask = years < test_year
    test_mask = years == test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  Train: {len(X_train):,} "
          f"({years[train_mask].min()}-{years[train_mask].max()})")
    print(f"  Test:  {len(X_test):,} ({test_year})")

    # Train
    print("\nTraining LightGBM...")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train,
                             feature_name=FEATURE_NAMES)
    valid_data = lgb.Dataset(X_test, label=y_test,
                             feature_name=FEATURE_NAMES,
                             reference=train_data)

    callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]
    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    # Evaluate
    preds = model.predict(X_test)

    # Brier score
    brier = float(np.mean((preds - y_test) ** 2))
    brier_baseline = float(np.mean((0.5 - y_test) ** 2))
    brier_skill = 1 - brier / brier_baseline

    # Log loss
    eps = 1e-7
    p_clip = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(
        y_test * np.log(p_clip) + (1 - y_test) * np.log(1 - p_clip)
    ))

    # ECE
    ece = 0.0
    calibration = []
    for low in np.arange(0, 1.0, 0.1):
        high = low + 0.1
        mask = (preds >= low) & (preds < high)
        if mask.sum() > 0:
            pred_mean = float(preds[mask].mean())
            act_mean = float(y_test[mask].mean())
            abs_err = abs(pred_mean - act_mean)
            ece += abs_err * mask.sum() / len(y_test)
            calibration.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "predicted": round(pred_mean, 4),
                "actual": round(act_mean, 4),
                "error": round(abs_err, 4),
                "n": int(mask.sum()),
            })

    # Results
    print(f"\n{'=' * 55}")
    print(f"LightGBM WP Model — Test Year: {test_year}")
    print(f"{'=' * 55}")
    print(f"  Brier Score:      {brier:.6f}")
    print(f"  Brier Skill:      {brier_skill:.4f}")
    print(f"  Log Loss:         {log_loss:.6f}")
    print(f"  ECE:              {ece:.4f}")
    print(f"  Best iteration:   {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    fi = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n  Feature Importance (gain):")
    for name, imp in fi:
        print(f"    {name:<20} {imp:>10.1f}")

    # Calibration
    print(f"\n  Calibration:")
    for c in calibration:
        diff = c["actual"] - c["predicted"]
        arrow = "^" if diff > 0.02 else "v" if diff < -0.02 else "="
        print(f"    {c['bin']}: pred={c['predicted']:.3f} "
              f"actual={c['actual']:.3f} {arrow} (n={c['n']:,})")

    # Save
    model_path = output_dir / "wp_lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\n  Model: {model_path}")

    metrics = {
        "test_year": test_year,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "brier_score": round(brier, 6),
        "brier_skill_score": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
        "best_iteration": model.best_iteration,
        "feature_importance": {n: round(float(v), 1) for n, v in fi},
        "calibration": calibration,
    }
    metrics_path = output_dir / "wp_lgbm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
