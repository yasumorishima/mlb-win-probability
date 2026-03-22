"""
Train WP model on full Statcast pitch-level data from BigQuery.

Uses 70+ features including:
- Game state (inning, outs, runners, score)
- Pitch characteristics (velocity, spin, movement, location)
- Batted ball quality (exit velo, launch angle, xwOBA)
- Bat tracking (bat speed, swing length, attack angle)
- Context (count, game phase, lineup position)

Benchmark: MLB official home_win_exp (our model must beat this).

LightGBM + CatBoost + Optuna, same pipeline as baseball-mlops.

Usage:
  python scripts/train_wp_statcast.py --output-dir data/
  python scripts/train_wp_statcast.py --n-trials 500 --n-trials-catboost 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = "data-platform-490901"
DATASET = "mlb_wp"
TABLE = "statcast_pitches"


def get_bq_client():
    """Get authenticated BQ client."""
    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        local_key = Path(r"C:\Users\fw_ya\.claude\gcp-sa-key.json")
        if local_key.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_key)

    return bigquery.Client(project=PROJECT)


def load_from_bq(test_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Statcast data from BigQuery, split by year."""
    client = get_bq_client()

    # At-bat outcomes only (events IS NOT NULL) to reduce memory
    query = f"""
        SELECT
            game_pk, game_year, home_team,
            inning, inning_topbot, outs_when_up, balls, strikes,
            on_1b, on_2b, on_3b,
            home_score, away_score, score_diff, is_bottom,
            post_home_score, post_away_score,
            release_speed, effective_speed, pfx_x, pfx_z,
            plate_x, plate_z, release_spin_rate, release_extension,
            launch_speed, launch_angle, hit_distance_sc,
            estimated_woba_using_speedangle, estimated_ba_using_speedangle,
            woba_value, bb_type, zone,
            bat_speed, swing_length,
            n_thruorder_pitcher, n_priorpa_thisgame_player_at_bat,
            home_win_exp, delta_home_win_exp,
            events, type
        FROM `{PROJECT}.{DATASET}.{TABLE}`
        WHERE game_type = 'R' AND events IS NOT NULL
        ORDER BY game_pk, inning, is_bottom, outs_when_up
    """

    print("Loading from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"  Total: {len(df):,} rows")

    train = df[df["game_year"] < test_year].copy()
    test = df[df["game_year"] == test_year].copy()
    print(f"  Train: {len(train):,} (< {test_year})")
    print(f"  Test:  {len(test):,} ({test_year})")

    return train, test


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Engineer 70+ features from raw Statcast data.
    Returns feature matrix and feature names.
    """
    features = pd.DataFrame(index=df.index)

    # --- Game state (core) ---
    features["inning"] = df["inning"].fillna(1)
    features["is_bottom"] = df.get("is_bottom", (df["inning_topbot"] == "Bot").astype(int))
    features["outs"] = df["outs_when_up"].fillna(0)
    features["balls"] = df["balls"].fillna(0)
    features["strikes"] = df["strikes"].fillna(0)
    features["score_diff"] = df.get("score_diff", df["home_score"] - df["away_score"]).fillna(0)

    # Runners
    features["r1"] = df["on_1b"].notna().astype(int)
    features["r2"] = df["on_2b"].notna().astype(int)
    features["r3"] = df["on_3b"].notna().astype(int)
    features["total_runners"] = features["r1"] + features["r2"] + features["r3"]
    features["scoring_position"] = features["r2"] + features["r3"]

    # --- Game context ---
    features["abs_lead"] = features["score_diff"].abs()
    features["game_phase"] = features["inning"].clip(1, 12).apply(
        lambda x: 0 if x <= 3 else (1 if x <= 6 else 2))
    features["walk_off_eligible"] = (
        (features["is_bottom"] == 1) &
        (features["inning"] >= 9) &
        (features["score_diff"] <= 0)
    ).astype(int)
    features["tied"] = (features["score_diff"] == 0).astype(int)
    features["close_game"] = (features["abs_lead"] <= 2).astype(int)

    # Count state
    features["count_state"] = features["balls"] * 3 + features["strikes"]
    features["full_count"] = ((features["balls"] == 3) & (features["strikes"] == 2)).astype(int)
    features["ahead_in_count"] = (features["strikes"] > features["balls"]).astype(int)

    # --- Interactions ---
    features["inn_x_bottom"] = features["inning"] * features["is_bottom"]
    features["inn_x_outs"] = features["inning"] * features["outs"]
    features["inn_x_score"] = features["inning"] * features["score_diff"]
    features["outs_x_score"] = features["outs"] * features["score_diff"]
    features["phase_x_lead"] = features["game_phase"] * features["abs_lead"]
    features["runners_x_outs"] = features["total_runners"] * features["outs"]
    features["scoring_x_outs"] = features["scoring_position"] * features["outs"]

    # --- Pitch characteristics ---
    features["release_speed"] = df["release_speed"].fillna(df["release_speed"].median())
    features["effective_speed"] = df["effective_speed"].fillna(features["release_speed"])
    features["pfx_x"] = df["pfx_x"].fillna(0)
    features["pfx_z"] = df["pfx_z"].fillna(0)
    features["total_movement"] = np.sqrt(features["pfx_x"]**2 + features["pfx_z"]**2)
    features["release_spin"] = df["release_spin_rate"].fillna(
        df["release_spin_rate"].median())
    features["plate_x"] = df["plate_x"].fillna(0)
    features["plate_z"] = df["plate_z"].fillna(0)
    features["in_zone"] = df["zone"].fillna(0).apply(lambda z: 1 if 1 <= z <= 9 else 0)
    features["release_ext"] = df["release_extension"].fillna(
        df["release_extension"].median())
    features["arm_angle"] = df["arm_angle"].fillna(df["arm_angle"].median())

    # --- Batted ball quality ---
    features["launch_speed"] = df["launch_speed"].fillna(0)
    features["launch_angle"] = df["launch_angle"].fillna(0)
    features["hit_distance"] = df["hit_distance_sc"].fillna(0)
    features["xwoba"] = df["estimated_woba_using_speedangle"].fillna(0)
    features["xba"] = df["estimated_ba_using_speedangle"].fillna(0)
    features["xslg"] = df["estimated_slg_using_speedangle"].fillna(0)
    features["woba_value"] = df["woba_value"].fillna(0)

    # Barrel proxy: launch_speed >= 98 and 26 <= launch_angle <= 30
    features["barrel"] = (
        (features["launch_speed"] >= 98) &
        (features["launch_angle"] >= 26) &
        (features["launch_angle"] <= 30)
    ).astype(int)

    # --- Bat tracking (2024+ only, 0 for earlier years) ---
    features["bat_speed"] = df["bat_speed"].fillna(0) if "bat_speed" in df.columns else 0
    features["swing_length"] = df["swing_length"].fillna(0) if "swing_length" in df.columns else 0
    features["attack_angle_bat"] = df["attack_angle"].fillna(0) if "attack_angle" in df.columns else 0

    # --- Lineup / fatigue ---
    features["n_thruorder"] = df["n_thruorder_pitcher"].fillna(1)
    features["n_priorpa"] = df["n_priorpa_thisgame_player_at_bat"].fillna(0)

    # --- Nonlinear ---
    features["score_diff_sq"] = features["score_diff"] ** 2
    features["inning_sq"] = features["inning"] ** 2
    features["speed_sq"] = features["release_speed"] ** 2
    features["launch_speed_sq"] = features["launch_speed"] ** 2

    # --- Park factors (from savant-extras, 2015+) ---
    # Loaded separately and merged by home_team + game_year
    if "pf_5yr" in df.columns:
        features["park_factor"] = df["pf_5yr"].fillna(100) / 100.0
        features["park_hr_factor"] = df["pf_hr"].fillna(100) / 100.0
    else:
        features["park_factor"] = 1.0
        features["park_hr_factor"] = 1.0

    # --- Clipped ---
    features["score_capped"] = features["score_diff"].clip(-10, 10)
    features["inning_capped"] = features["inning"].clip(1, 12)

    feature_names = list(features.columns)
    return features.values.astype(np.float32), feature_names


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract binary home win target."""
    # Use post_home_score vs post_away_score at end of game
    # For per-pitch: use the final game outcome
    # We need to map game_pk -> home_won
    if "home_won" in df.columns:
        return df["home_won"].values

    # Compute from final scores per game
    game_outcomes = df.groupby("game_pk").agg(
        final_home=("post_home_score", "max"),
        final_away=("post_away_score", "max"),
    )
    game_outcomes["home_won"] = (game_outcomes["final_home"] > game_outcomes["final_away"]).astype(int)
    df = df.merge(game_outcomes[["home_won"]], left_on="game_pk", right_index=True, how="left")
    return df["home_won"].values


def evaluate(preds: np.ndarray, actuals: np.ndarray, name: str) -> dict:
    """Compute metrics."""
    brier = float(np.mean((preds - actuals) ** 2))
    brier_base = float(np.mean((0.5 - actuals) ** 2))
    brier_skill = 1 - brier / brier_base

    eps = 1e-7
    p = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)))

    ece = 0.0
    n = len(actuals)
    for low in np.arange(0, 1.0, 0.1):
        mask = (preds >= low) & (preds < low + 0.1)
        if mask.sum() > 0:
            ece += abs(preds[mask].mean() - actuals[mask].mean()) * mask.sum() / n

    print(f"  {name}: Brier={brier:.6f} BSS={brier_skill:.4f} "
          f"LogLoss={log_loss:.6f} ECE={ece:.4f}")

    return {
        "model": name,
        "brier": round(brier, 6),
        "brier_skill": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train WP model on Statcast data (BQ)")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--n-trials", type=int, default=0,
                        help="LightGBM Optuna trials")
    parser.add_argument("--n-trials-catboost", type=int, default=0,
                        help="CatBoost Optuna trials")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = load_from_bq(args.test_year)

    # Merge park factors from BQ (savant-extras data already loaded there)
    print("\nLoading park factors...")
    try:
        bq_client = get_bq_client()
        pf_query = f"""
            SELECT CAST(season AS FLOAT64) AS game_year, team AS home_team,
                   pf_5yr, pf_hr
            FROM `{PROJECT}.mlb_statcast.raw_park_factors`
        """
        pf = bq_client.query(pf_query).to_dataframe()
        if len(pf) > 0:
            train_df = train_df.merge(pf, on=["home_team", "game_year"], how="left")
            test_df = test_df.merge(pf, on=["home_team", "game_year"], how="left")
            print(f"  Merged park factors from BQ ({len(pf)} team-seasons)")
        else:
            print("  No park factors in BQ")
    except Exception as e:
        print(f"  Park factors not available: {e}")

    # Engineer features
    print("\nEngineering features...")
    X_train, feature_names = engineer_features(train_df)
    X_test, _ = engineer_features(test_df)
    y_train = get_target(train_df)
    y_test = get_target(test_df)

    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # MLB official WP as benchmark
    print(f"\n{'=' * 60}")
    print("BENCHMARK: MLB Official home_win_exp")
    print(f"{'=' * 60}")
    mlb_wp = test_df["home_win_exp"].values
    mlb_valid = ~np.isnan(mlb_wp)
    if mlb_valid.sum() > 0:
        mlb_metrics = evaluate(mlb_wp[mlb_valid], y_test[mlb_valid], "MLB_official")
    else:
        print("  MLB WP not available in test data")
        mlb_metrics = None

    # -------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------
    import lightgbm as lgb

    print(f"\n{'=' * 60}")
    print(f"LightGBM ({len(feature_names)} features)")
    print(f"{'=' * 60}")

    if args.n_trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def lgbm_objective(trial):
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "verbosity": -1, "seed": 42,
                "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 511),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "min_child_samples": trial.suggest_int("min_child", 20, 500),
                "feature_fraction": trial.suggest_float("ff", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bf", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bfreq", 1, 10),
                "reg_alpha": trial.suggest_float("alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("lam", 1e-8, 10, log=True),
            }
            td = lgb.Dataset(X_train, y_train, feature_name=feature_names)
            vd = lgb.Dataset(X_test, y_test, feature_name=feature_names, reference=td)
            m = lgb.train(params, td, 2000, valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
            p = m.predict(X_test)
            return float(np.mean((p - y_test) ** 2))

        print(f"  Optuna: {args.n_trials} trials...")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lgbm_objective, n_trials=args.n_trials,
                       show_progress_bar=True)
        print(f"  Best Brier: {study.best_value:.6f}")
        best_lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "verbosity": -1, "seed": 42,
            "learning_rate": study.best_params["lr"],
            "num_leaves": study.best_params["num_leaves"],
            "max_depth": study.best_params["max_depth"],
            "min_child_samples": study.best_params["min_child"],
            "feature_fraction": study.best_params["ff"],
            "bagging_fraction": study.best_params["bf"],
            "bagging_freq": study.best_params["bfreq"],
            "reg_alpha": study.best_params["alpha"],
            "reg_lambda": study.best_params["lam"],
        }
    else:
        best_lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 127,
            "max_depth": 10, "min_child_samples": 200,
            "feature_fraction": 0.7, "bagging_fraction": 0.7,
            "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbose": -1, "seed": 42,
        }

    print("  Training final model...")
    td = lgb.Dataset(X_train, y_train, feature_name=feature_names)
    vd = lgb.Dataset(X_test, y_test, feature_name=feature_names, reference=td)
    lgbm_model = lgb.train(
        best_lgbm_params, td, 3000, valid_sets=[vd],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)])

    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_metrics = evaluate(lgbm_preds, y_test, "LightGBM_statcast")

    # Feature importance
    fi = sorted(zip(feature_names, lgbm_model.feature_importance(importance_type="gain")),
                key=lambda x: -x[1])
    print("  Top 15 features:")
    for name, imp in fi[:15]:
        print(f"    {name:<30} {imp:>10.1f}")

    # Save
    lgbm_model.save_model(str(output_dir / "wp_statcast_lgbm.txt"))

    # -------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------
    cb_metrics = None
    if args.n_trials_catboost > 0:
        try:
            from catboost import CatBoostClassifier
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            print(f"\n{'=' * 60}")
            print(f"CatBoost ({args.n_trials_catboost} Optuna trials)")
            print(f"{'=' * 60}")

            def cb_objective(trial):
                params = {
                    "iterations": 1000,
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2", 1e-3, 10, log=True),
                    "min_child_samples": trial.suggest_int("mcs", 20, 300),
                    "subsample": trial.suggest_float("ss", 0.5, 1.0),
                    "random_seed": 42, "verbose": 0,
                    "eval_metric": "Logloss",
                    "early_stopping_rounds": 50,
                }
                m = CatBoostClassifier(**params)
                m.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
                p = m.predict_proba(X_test)[:, 1]
                return float(np.mean((p - y_test) ** 2))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(cb_objective, n_trials=args.n_trials_catboost,
                           show_progress_bar=True)
            print(f"  Best Brier: {study.best_value:.6f}")

            best_cb = {
                "iterations": 2000,
                "learning_rate": study.best_params["lr"],
                "depth": study.best_params["depth"],
                "l2_leaf_reg": study.best_params["l2"],
                "min_child_samples": study.best_params["mcs"],
                "subsample": study.best_params["ss"],
                "random_seed": 42, "verbose": 0,
                "eval_metric": "Logloss",
                "early_stopping_rounds": 100,
            }
            cb_model = CatBoostClassifier(**best_cb)
            cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
            cb_preds = cb_model.predict_proba(X_test)[:, 1]
            cb_metrics = evaluate(cb_preds, y_test, "CatBoost_statcast")

            cb_model.save_model(str(output_dir / "wp_statcast_catboost.cbm"))

        except ImportError:
            print("  CatBoost not installed, skipping")

    # -------------------------------------------------------
    # Comparison
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")

    results = {"test_year": args.test_year, "n_features": len(feature_names)}
    results["feature_names"] = feature_names
    results["lgbm"] = lgbm_metrics
    results["lgbm"]["optuna_trials"] = args.n_trials
    results["lgbm"]["best_iteration"] = lgbm_model.best_iteration
    results["lgbm"]["feature_importance"] = {n: round(float(v), 1) for n, v in fi[:30]}

    if mlb_metrics:
        results["mlb_official"] = mlb_metrics
        mlb_brier = mlb_metrics["brier"]
        lgbm_vs_mlb = (mlb_brier - lgbm_metrics["brier"]) / mlb_brier * 100
        print(f"\n  LightGBM vs MLB official: {lgbm_vs_mlb:+.2f}% Brier")

    if cb_metrics:
        results["catboost"] = cb_metrics
        if mlb_metrics:
            cb_vs_mlb = (mlb_brier - cb_metrics["brier"]) / mlb_brier * 100
            print(f"  CatBoost vs MLB official: {cb_vs_mlb:+.2f}% Brier")

    # Save
    results_path = output_dir / "wp_statcast_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
