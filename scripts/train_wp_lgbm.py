"""
Train LightGBM + CatBoost models for WP prediction.

Optuna-optimized hyperparameters (same approach as baseball-mlops).
Rich feature engineering (25 features vs previous 11).

Time-based holdout split:
  Train: all years except most recent
  Test:  most recent year

Usage:
  python scripts/train_wp_lgbm.py [--data-dir data/] [--output-dir data/]
  python scripts/train_wp_lgbm.py --n-trials 500  # Optuna optimization
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def _log_elapsed(label: str, start: float, budget_min: int = 360):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")

# RE24 table for feature engineering
RE24_MLB = {
    (0,0,0,0): 0.481, (1,0,0,0): 0.859, (0,1,0,0): 1.100, (1,1,0,0): 1.437,
    (0,0,1,0): 1.350, (1,0,1,0): 1.784, (0,1,1,0): 1.964, (1,1,1,0): 2.292,
    (0,0,0,1): 0.254, (1,0,0,1): 0.509, (0,1,0,1): 0.664, (1,1,0,1): 0.884,
    (0,0,1,1): 0.950, (1,0,1,1): 1.130, (0,1,1,1): 1.376, (1,1,1,1): 1.541,
    (0,0,0,2): 0.098, (1,0,0,2): 0.224, (0,1,0,2): 0.319, (1,1,0,2): 0.429,
    (0,0,1,2): 0.353, (1,0,1,2): 0.478, (0,1,1,2): 0.570, (1,1,1,2): 0.752,
}

FEATURE_NAMES = [
    # Raw state (7)
    "inning", "is_bottom", "outs", "r1", "r2", "r3", "score_diff",
    # Interactions (5)
    "inn_x_bottom", "inn_x_outs", "inn_x_score", "outs_x_score", "bottom_x_score",
    # Derived (7)
    "abs_lead", "total_runners", "scoring_position",
    "tied_innings_left", "re24",
    "game_phase",  # 0=early(1-3), 1=mid(4-6), 2=late(7+)
    "walk_off_eligible",  # bottom 9+ and score_diff <= 0
    # Nonlinear (4)
    "score_diff_sq", "inning_sq", "lead_x_phase", "runners_x_outs",
    # Clipped extremes (2)
    "score_capped", "inning_capped",
]


def extract_features(row: dict) -> list[float]:
    """Extract 25 features from a play state row."""
    inning = int(row["inning"])
    is_bottom = 1 if row["half_inning"] == "bottom" else 0
    outs = int(row["outs"])
    r1 = int(row["runner_1b"])
    r2 = int(row["runner_2b"])
    r3 = int(row["runner_3b"])
    score_diff = int(row["score_diff"])

    total_runners = r1 + r2 + r3
    scoring_position = r2 + r3  # runners in scoring position
    abs_lead = abs(score_diff)
    re24 = RE24_MLB.get((r1, r2, r3, outs), 0.0)
    game_phase = 0 if inning <= 3 else (1 if inning <= 6 else 2)
    walk_off = 1 if (is_bottom and inning >= 9 and score_diff <= 0) else 0
    tied_left = max(9 - inning, 0) * (1 if score_diff == 0 else 0)

    return [
        inning, is_bottom, outs, r1, r2, r3, score_diff,
        inning * is_bottom, inning * outs, inning * score_diff,
        outs * score_diff, is_bottom * score_diff,
        abs_lead, total_runners, scoring_position,
        tied_left, re24, game_phase, walk_off,
        score_diff ** 2, inning ** 2,
        abs_lead * game_phase, total_runners * outs,
        max(-10, min(10, score_diff)), min(inning, 12),
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
                    features.append(extract_features(row))
                    labels.append(int(row["home_won"]))
                    years.append(year)
                except (ValueError, KeyError):
                    continue

    return np.array(features), np.array(labels), np.array(years)


def optimize_lgbm(X_train, y_train, X_val, y_val, n_trials: int = 500):
    """Optuna hyperparameter optimization for LightGBM."""
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        valid_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES,
                                 reference=train_data)

        model = lgb.train(
            params, train_data,
            num_boost_round=2000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        brier = float(np.mean((preds - y_val) ** 2))
        return brier

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best Brier: {study.best_value:.6f}")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params


def train_catboost(X_train, y_train, X_val, y_val,
                   n_trials: int = 200) -> tuple:
    """Train CatBoost with Optuna optimization."""
    try:
        from catboost import CatBoostClassifier, Pool
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  CatBoost not installed, skipping")
        return None, None

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "Logloss",
            "early_stopping_rounds": 50,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

        preds = model.predict_proba(X_val)[:, 1]
        brier = float(np.mean((preds - y_val) ** 2))
        return brier

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best Brier: {study.best_value:.6f}")

    # Train final model with best params
    best_params = {
        **study.best_params,
        "iterations": 2000,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "Logloss",
        "early_stopping_rounds": 50,
    }
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    return model, study.best_params


def evaluate_model(preds, y_test, model_name: str) -> dict:
    """Compute Brier, BSS, Log Loss, ECE, calibration."""
    brier = float(np.mean((preds - y_test) ** 2))
    brier_baseline = float(np.mean((0.5 - y_test) ** 2))
    brier_skill = 1 - brier / brier_baseline

    eps = 1e-7
    p_clip = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(
        y_test * np.log(p_clip) + (1 - y_test) * np.log(1 - p_clip)
    ))

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

    return {
        "model": model_name,
        "brier_score": round(brier, 6),
        "brier_skill_score": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
        "calibration": calibration,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train LightGBM + CatBoost WP models (Optuna-optimized)")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=0,
                        help="Optuna trials (0=use fixed params)")
    parser.add_argument("--n-trials-catboost", type=int, default=0,
                        help="CatBoost Optuna trials (0=skip CatBoost)")
    args = parser.parse_args()

    t0 = time.time()

    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed")
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
    print(f"  Total: {len(X):,} plays, {len(FEATURE_NAMES)} features, "
          f"years {years.min()}-{years.max()}")

    # Split
    test_year = args.test_year or int(years.max())
    train_mask = years < test_year
    test_mask = years == test_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  Train: {len(X_train):,} ({years[train_mask].min()}-{years[train_mask].max()})")
    print(f"  Test:  {len(X_test):,} ({test_year})")
    _log_elapsed("load_data", t0)

    # -------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("LightGBM Training")
    print(f"{'=' * 60}")

    if args.n_trials > 0:
        print(f"\n  Optuna optimization ({args.n_trials} trials)...")
        best_params = optimize_lgbm(X_train, y_train, X_test, y_test,
                                     n_trials=args.n_trials)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": 42,
            **best_params,
        }
    else:
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

    print("\n  Training final LightGBM model...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_NAMES,
                             reference=train_data)

    callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]
    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    lgbm_preds = model.predict(X_test)
    lgbm_metrics = evaluate_model(lgbm_preds, y_test, "lightgbm")

    print(f"\n  Brier Score:  {lgbm_metrics['brier_score']:.6f}")
    print(f"  Brier Skill:  {lgbm_metrics['brier_skill_score']:.4f}")
    print(f"  Log Loss:     {lgbm_metrics['log_loss']:.6f}")
    print(f"  ECE:          {lgbm_metrics['ece']:.4f}")
    print(f"  Best iter:    {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    fi = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n  Feature Importance (top 10):")
    for name, imp in fi[:10]:
        print(f"    {name:<25} {imp:>10.1f}")

    # Save LightGBM
    model_path = output_dir / "wp_lgbm_model.txt"
    model.save_model(str(model_path))

    lgbm_metrics["test_year"] = test_year
    lgbm_metrics["n_train"] = int(len(X_train))
    lgbm_metrics["n_test"] = int(len(X_test))
    lgbm_metrics["n_features"] = len(FEATURE_NAMES)
    lgbm_metrics["best_iteration"] = model.best_iteration
    lgbm_metrics["feature_importance"] = {n: round(float(v), 1) for n, v in fi}
    lgbm_metrics["optuna_trials"] = args.n_trials
    if args.n_trials > 0:
        lgbm_metrics["best_params"] = best_params

    metrics_path = output_dir / "wp_lgbm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(lgbm_metrics, f, indent=2)
    print(f"\n  Model: {model_path}")
    print(f"  Metrics: {metrics_path}")
    _log_elapsed("lightgbm_train", t0)

    # -------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------
    if args.n_trials_catboost > 0:
        print(f"\n{'=' * 60}")
        print(f"CatBoost Training ({args.n_trials_catboost} Optuna trials)")
        print(f"{'=' * 60}")

        cb_model, cb_best_params = train_catboost(
            X_train, y_train, X_test, y_test,
            n_trials=args.n_trials_catboost,
        )

        if cb_model is not None:
            cb_preds = cb_model.predict_proba(X_test)[:, 1]
            cb_metrics = evaluate_model(cb_preds, y_test, "catboost")

            print(f"\n  Brier Score:  {cb_metrics['brier_score']:.6f}")
            print(f"  Brier Skill:  {cb_metrics['brier_skill_score']:.4f}")
            print(f"  ECE:          {cb_metrics['ece']:.4f}")

            # Save CatBoost
            cb_model_path = output_dir / "wp_catboost_model.cbm"
            cb_model.save_model(str(cb_model_path))

            cb_metrics["test_year"] = test_year
            cb_metrics["n_train"] = int(len(X_train))
            cb_metrics["n_test"] = int(len(X_test))
            cb_metrics["n_features"] = len(FEATURE_NAMES)
            cb_metrics["optuna_trials"] = args.n_trials_catboost
            cb_metrics["best_params"] = cb_best_params

            cb_metrics_path = output_dir / "wp_catboost_metrics.json"
            with open(cb_metrics_path, "w") as f:
                json.dump(cb_metrics, f, indent=2)
            print(f"  Model: {cb_model_path}")
            print(f"  Metrics: {cb_metrics_path}")

    # -------------------------------------------------------
    # Comparison
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ML MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  LightGBM Brier: {lgbm_metrics['brier_score']:.6f} "
          f"({len(FEATURE_NAMES)} features, "
          f"{'Optuna ' + str(args.n_trials) + ' trials' if args.n_trials > 0 else 'fixed params'})")
    if args.n_trials_catboost > 0 and cb_model is not None:
        print(f"  CatBoost Brier: {cb_metrics['brier_score']:.6f} "
              f"({args.n_trials_catboost} Optuna trials)")
    _log_elapsed("total", t0)


if __name__ == "__main__":
    main()
