"""
Ensemble WP Engine — Combine v1/v2/LightGBM with inverse-Brier weighting.

Same philosophy as baseball-mlops 5-model ensemble:
  weight_i = 1 / brier_i
  ensemble = sum(w_i * pred_i) / sum(w_i)

Additionally applies Isotonic Regression calibration to correct ECE.

Outputs:
  - data/ensemble_weights.json  — per-engine weights from Brier scores
  - data/calibrator.pkl         — trained IsotonicRegression model

Usage:
  python scripts/ensemble_wp.py --data-dir data/ --output-dir data/
  python scripts/ensemble_wp.py --data-dir data/ --output-dir data/ --test-year 2024
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def _log_elapsed(label: str, start: float, budget_min: int = 120):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")

sys.path.insert(0, str(Path(__file__).parent.parent))


FEATURE_NAMES = [
    "inning", "is_bottom", "outs", "r1", "r2", "r3", "score_diff",
    "inn_x_bottom", "abs_lead", "total_runners", "tied_innings_left",
]


def load_play_states(csv_path: Path) -> list[dict]:
    """Load play states from a CSV file."""
    states = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                states.append({
                    "game_pk": int(row.get("game_pk", 0)),
                    "inning": int(row["inning"]),
                    "half_inning": row["half_inning"],
                    "outs": int(row["outs"]),
                    "runners": (int(row["runner_1b"]),
                                int(row["runner_2b"]),
                                int(row["runner_3b"])),
                    "score_diff": int(row["score_diff"]),
                    "home_won": int(row["home_won"]),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                })
            except (ValueError, KeyError):
                continue
    return states


def predict_v1(states: list[dict]) -> np.ndarray:
    """Generate predictions using v1 (Normal approximation)."""
    from win_probability import calculate_wp, MLB_RPG
    return np.array([
        calculate_wp(s["inning"], s["half_inning"], s["outs"],
                     s["runners"], s["score_diff"], MLB_RPG)
        for s in states
    ])


def predict_v2(states: list[dict], data_dir: Path,
               exclude_year: int | None = None) -> np.ndarray | None:
    """Generate predictions using v2 (Empirical + Markov).

    If exclude_year is set, builds a fair table excluding that year.
    """
    from win_probability import MLB_RPG

    try:
        if exclude_year:
            from scripts.build_wp_v2 import (
                load_all_play_states,
                extract_transitions,
                build_transition_matrix,
                simulate_run_distributions,
                build_empirical_wp,
            )
            from win_probability_v2 import WPEngineV2

            train_states = load_all_play_states(data_dir, {exclude_year})
            if not train_states:
                return None
            transitions = extract_transitions(train_states)
            trans_probs = build_transition_matrix(transitions)
            run_dists = simulate_run_distributions(trans_probs, n_sims=200_000)
            wp_table = build_empirical_wp(train_states, run_dists)

            engine = WPEngineV2.__new__(WPEngineV2)
            engine.wp_table = wp_table
            engine.run_dists = run_dists
            engine._loaded = True
        else:
            from win_probability_v2 import WPEngineV2
            engine = WPEngineV2(data_dir)
            if not engine.is_loaded:
                return None

        return np.array([
            engine.calculate_wp(s["inning"], s["half_inning"], s["outs"],
                                s["runners"], s["score_diff"], MLB_RPG)
            for s in states
        ])
    except Exception as e:
        print(f"  v2 prediction failed: {e}")
        return None


def _states_to_rows(states: list[dict]) -> list[dict]:
    """Convert internal state dicts to CSV-row-like dicts for extract_features."""
    rows = []
    for s in states:
        rows.append({
            "inning": str(s["inning"]),
            "half_inning": s["half_inning"],
            "outs": str(s["outs"]),
            "runner_1b": str(s["runners"][0]),
            "runner_2b": str(s["runners"][1]),
            "runner_3b": str(s["runners"][2]),
            "score_diff": str(s["score_diff"]),
            "home_won": str(s["home_won"]),
        })
    return rows


def predict_lgbm(states: list[dict], data_dir: Path) -> np.ndarray | None:
    """Generate predictions using LightGBM model."""
    lgbm_path = data_dir / "wp_lgbm_model.txt"
    if not lgbm_path.exists():
        return None
    try:
        import lightgbm as lgb
        from scripts.train_wp_lgbm import extract_features

        model = lgb.Booster(model_file=str(lgbm_path))
        rows = _states_to_rows(states)
        X = np.array([extract_features(r) for r in rows])
        return model.predict(X)
    except ImportError:
        print("  LightGBM not installed")
        return None


def predict_catboost(states: list[dict], data_dir: Path) -> np.ndarray | None:
    """Generate predictions using CatBoost model."""
    cb_path = data_dir / "wp_catboost_model.cbm"
    if not cb_path.exists():
        return None
    try:
        from catboost import CatBoostClassifier
        from scripts.train_wp_lgbm import extract_features

        model = CatBoostClassifier()
        model.load_model(str(cb_path))
        rows = _states_to_rows(states)
        X = np.array([extract_features(r) for r in rows])
        return model.predict_proba(X)[:, 1]
    except ImportError:
        print("  CatBoost not installed")
        return None


def _load_statcast_csvs(data_dir: Path, years: list[int] | None = None,
                        ) -> "pd.DataFrame | None":
    """Load statcast_pitches CSVs as a pandas DataFrame.

    Returns None if no statcast CSVs found.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    csvs = sorted(data_dir.glob("statcast_pitches_*.csv"))
    if not csvs:
        return None

    dfs = []
    for c in csvs:
        year = int(float(c.stem.split("_")[-1]))
        if years and year not in years:
            continue
        dfs.append(pd.read_csv(c))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} at-bat outcomes from statcast CSVs")
    return df


def _build_statcast_lookup(df: "pd.DataFrame", model_path: Path,
                           ) -> dict[tuple, float]:
    """Build a lookup of game state -> WP prediction with real Statcast features.

    Key: (game_pk, inning, is_bottom, outs, score_diff, r1, r2, r3)
    Including runners makes the key nearly unique per at-bat within a game.

    Uses engineer_features() from train_wp_statcast to build real features,
    then predicts with the Statcast LightGBM model.
    """
    import lightgbm as lgb
    import pandas as pd
    from scripts.train_wp_statcast import engineer_features

    model = lgb.Booster(model_file=str(model_path))
    X, feature_names = engineer_features(df)
    preds = np.clip(model.predict(X), 0.001, 0.999)

    is_bottom = (
        df["is_bottom"].values if "is_bottom" in df.columns
        else (df["inning_topbot"] == "Bot").astype(int).values
    )
    score_diff = (
        df["score_diff"].values if "score_diff" in df.columns
        else (df["home_score"] - df["away_score"]).values
    )
    # Convert runner IDs (or NaN) to 0/1
    r1 = df["on_1b"].notna().astype(int).values if "on_1b" in df.columns else np.zeros(len(df), dtype=int)
    r2 = df["on_2b"].notna().astype(int).values if "on_2b" in df.columns else np.zeros(len(df), dtype=int)
    r3 = df["on_3b"].notna().astype(int).values if "on_3b" in df.columns else np.zeros(len(df), dtype=int)

    lookup = {}
    for i in range(len(df)):
        key = (
            int(df["game_pk"].iloc[i]),
            int(df["inning"].iloc[i]),
            int(is_bottom[i]),
            int(df["outs_when_up"].iloc[i]),
            int(score_diff[i]),
            int(r1[i]), int(r2[i]), int(r3[i]),
        )
        lookup[key] = float(preds[i])

    n_collisions = len(df) - len(lookup)
    print(f"  Statcast lookup: {len(lookup):,} unique states "
          f"({n_collisions:,} collisions), mean WP={np.mean(preds):.4f}")
    return lookup


def predict_statcast(states: list[dict], data_dir: Path) -> np.ndarray | None:
    """Generate predictions using Statcast LightGBM model.

    If statcast_pitches CSVs are available, uses real pitch/hit features
    for at-bat plays (joined by game state). Falls back to game-state-only
    prediction for non-AB plays.
    """
    model_path = data_dir / "wp_statcast_lgbm.txt"
    if not model_path.exists():
        return None
    try:
        import lightgbm as lgb
        from win_probability_statcast import WPEngineStatcast

        engine = WPEngineStatcast()
        if not engine.is_loaded:
            return None

        # Try to load real statcast features
        statcast_df = _load_statcast_csvs(data_dir)
        lookup = None
        if statcast_df is not None:
            lookup = _build_statcast_lookup(statcast_df, model_path)

        # Predict for each play state
        preds = np.zeros(len(states))
        n_matched = 0
        for i, s in enumerate(states):
            is_bottom = 1 if s["half_inning"] == "bottom" else 0
            r1, r2, r3 = s["runners"]
            game_pk = s.get("game_pk", 0)
            key = (game_pk, s["inning"], is_bottom, s["outs"],
                   s["score_diff"], r1, r2, r3)

            if lookup and key in lookup:
                preds[i] = lookup[key]
                n_matched += 1
            else:
                # Fallback: game-state-only prediction
                gs = {
                    "inning": s["inning"],
                    "top_bottom": "bottom" if is_bottom else "top",
                    "outs": s["outs"],
                    "runners": s["runners"],
                    "score_diff": s["score_diff"],
                    "balls": 0, "strikes": 0,
                }
                feat = engine._build_features(gs, None, None)
                preds[i] = engine._model.predict(
                    np.array([feat], dtype=np.float32))[0]

        preds = np.clip(preds, 0.001, 0.999)

        if lookup:
            pct = n_matched / len(states) * 100 if states else 0
            print(f"  Statcast: {n_matched:,}/{len(states):,} matched "
                  f"({pct:.1f}%) with real features")
        else:
            print("  (game-state only — no statcast CSVs found)")

        return preds

    except ImportError as e:
        print(f"  LightGBM/pandas not installed (for Statcast): {e}")
        return None
    except Exception as e:
        print(f"  Statcast prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_bayesian(states: list[dict], data_dir: Path,
                     ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Generate predictions using Bayesian hierarchical model.

    Returns: (mean_wp, lower_5, upper_95) or (None, None, None).
    """
    posterior_path = data_dir / "bayesian_posterior.json"
    if not posterior_path.exists():
        return None, None, None
    try:
        from win_probability_bayesian import WPEngineBayesian

        engine = WPEngineBayesian(data_dir)
        if not engine.is_loaded:
            return None, None, None

        mean_preds = np.zeros(len(states))
        lower_preds = np.zeros(len(states))
        upper_preds = np.zeros(len(states))

        for i, s in enumerate(states):
            # States have home_team/away_team as full names, need abbreviation
            from scripts.train_wp_bayesian import TEAM_ABBREV
            home = TEAM_ABBREV.get(
                s.get("home_team", ""), s.get("home_team", ""))
            away = TEAM_ABBREV.get(
                s.get("away_team", ""), s.get("away_team", ""))

            result = engine.calculate_wp_with_ci(
                s["inning"], s["half_inning"], s["outs"],
                s["runners"], s["score_diff"],
                home_team=home, away_team=away,
                n_samples=100)

            mean_preds[i] = result.wp
            lower_preds[i] = result.wp_lower
            upper_preds[i] = result.wp_upper

        return mean_preds, lower_preds, upper_preds

    except Exception as e:
        print(f"  Bayesian prediction failed: {e}")
        return None, None, None


def compute_brier(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(np.mean((preds - actuals) ** 2))


def compute_ece(preds: np.ndarray, actuals: np.ndarray,
                n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    ece = 0.0
    n = len(actuals)
    for low in np.linspace(0, 1 - 1/n_bins, n_bins):
        high = low + 1/n_bins
        mask = (preds >= low) & (preds < high)
        if mask.sum() > 0:
            ece += abs(preds[mask].mean() - actuals[mask].mean()) * mask.sum() / n
    return ece


def ensemble_predictions(
    engine_preds: dict[str, np.ndarray],
    engine_briers: dict[str, float],
) -> np.ndarray:
    """Combine engine predictions using inverse-Brier weighting."""
    weights = {}
    for name, brier in engine_briers.items():
        if brier > 0:
            weights[name] = 1.0 / brier

    total_weight = sum(weights.values())
    if total_weight == 0:
        # Fallback: equal weights
        n = len(engine_preds)
        return sum(engine_preds.values()) / n

    result = np.zeros_like(list(engine_preds.values())[0])
    for name, preds in engine_preds.items():
        w = weights.get(name, 0) / total_weight
        result += w * preds

    return np.clip(result, 0.01, 0.99)


def calibrate_predictions(
    train_preds: np.ndarray,
    train_actuals: np.ndarray,
    test_preds: np.ndarray,
) -> np.ndarray:
    """Apply Isotonic Regression calibration."""
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    ir.fit(train_preds, train_actuals)
    return ir.predict(test_preds)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ensemble WP — inverse-Brier weighted + calibration")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run leave-one-year-out cross-validation")
    args = parser.parse_args()

    t0 = time.time()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine available years
    csvs = sorted(data_dir.glob("play_states_*.csv"))
    if not csvs:
        print("ERROR: No play_states CSVs found")
        sys.exit(1)

    all_years = [int(c.stem.split("_")[-1]) for c in csvs]
    print(f"Available years: {all_years}")

    if args.cross_validate:
        # -------------------------------------------------------
        # Leave-one-year-out cross-validation
        # -------------------------------------------------------
        print(f"\n{'=' * 70}")
        print("LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
        print(f"{'=' * 70}")

        cv_results = []
        for holdout_year in all_years:
            print(f"\n--- Holdout: {holdout_year} ---")

            # Load holdout data
            holdout_path = data_dir / f"play_states_{holdout_year}.csv"
            test_states = load_play_states(holdout_path)
            actuals = np.array([s["home_won"] for s in test_states], dtype=float)

            # Load train data (all other years)
            train_states = []
            for year in all_years:
                if year != holdout_year:
                    train_states.extend(
                        load_play_states(data_dir / f"play_states_{year}.csv"))
            train_actuals = np.array(
                [s["home_won"] for s in train_states], dtype=float)

            # Predictions
            engine_preds = {}
            engine_briers = {}

            v1_preds = predict_v1(test_states)
            engine_preds["v1"] = v1_preds
            engine_briers["v1"] = compute_brier(v1_preds, actuals)

            v2_preds = predict_v2(test_states, data_dir,
                                  exclude_year=holdout_year)
            if v2_preds is not None:
                engine_preds["v2"] = v2_preds
                engine_briers["v2"] = compute_brier(v2_preds, actuals)

            lgbm_preds = predict_lgbm(test_states, data_dir)
            if lgbm_preds is not None:
                engine_preds["lgbm"] = lgbm_preds
                engine_briers["lgbm"] = compute_brier(lgbm_preds, actuals)

            statcast_preds = predict_statcast(test_states, data_dir)
            if statcast_preds is not None:
                engine_preds["statcast"] = statcast_preds
                engine_briers["statcast"] = compute_brier(statcast_preds, actuals)

            # Ensemble (raw)
            ens_preds = ensemble_predictions(engine_preds, engine_briers)
            ens_brier = compute_brier(ens_preds, actuals)
            ens_ece = compute_ece(ens_preds, actuals)

            # Calibrated ensemble
            train_v1 = predict_v1(train_states)
            cal_preds = calibrate_predictions(train_v1, train_actuals,
                                              ens_preds)
            cal_brier = compute_brier(cal_preds, actuals)
            cal_ece = compute_ece(cal_preds, actuals)

            result = {
                "year": holdout_year,
                "n_plays": len(test_states),
                "engines": engine_briers,
                "ensemble_brier": round(ens_brier, 6),
                "ensemble_ece": round(ens_ece, 4),
                "calibrated_brier": round(cal_brier, 6),
                "calibrated_ece": round(cal_ece, 4),
            }
            cv_results.append(result)

            print(f"  v1 Brier: {engine_briers['v1']:.6f}", end="")
            if "v2" in engine_briers:
                print(f" | v2: {engine_briers['v2']:.6f}", end="")
            if "lgbm" in engine_briers:
                print(f" | lgbm: {engine_briers['lgbm']:.6f}", end="")
            print(f" | ensemble: {ens_brier:.6f} | calibrated: {cal_brier:.6f}")

        # Summary
        print(f"\n{'=' * 70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Year':<8} {'v1':>10} {'v2':>10} {'lgbm':>10} "
              f"{'statcast':>10} {'ensemble':>10} {'calibrated':>10}")
        print("-" * 70)
        for r in cv_results:
            row = f"{r['year']:<8} {r['engines']['v1']:>10.6f}"
            row += f" {r['engines'].get('v2', float('nan')):>10.6f}"
            row += f" {r['engines'].get('lgbm', float('nan')):>10.6f}"
            row += f" {r['engines'].get('statcast', float('nan')):>10.6f}"
            row += f" {r['ensemble_brier']:>10.6f}"
            row += f" {r['calibrated_brier']:>10.6f}"
            print(row)

        # Averages
        avg_v1 = np.mean([r["engines"]["v1"] for r in cv_results])
        avg_ens = np.mean([r["ensemble_brier"] for r in cv_results])
        avg_cal = np.mean([r["calibrated_brier"] for r in cv_results])
        print("-" * 60)
        print(f"{'Mean':<8} {avg_v1:>10.6f} {'':>10} {'':>10} "
              f"{avg_ens:>10.6f} {avg_cal:>10.6f}")

        imp_ens = (avg_v1 - avg_ens) / avg_v1 * 100
        imp_cal = (avg_v1 - avg_cal) / avg_v1 * 100
        print(f"\n  Ensemble vs v1: {imp_ens:+.2f}%")
        print(f"  Calibrated vs v1: {imp_cal:+.2f}%")

        # Save
        cv_path = output_dir / "ensemble_cv_results.json"
        with open(cv_path, "w") as f:
            json.dump({
                "years": all_years,
                "results": cv_results,
                "mean_v1_brier": round(avg_v1, 6),
                "mean_ensemble_brier": round(avg_ens, 6),
                "mean_calibrated_brier": round(avg_cal, 6),
                "improvement_ensemble_pct": round(imp_ens, 2),
                "improvement_calibrated_pct": round(imp_cal, 2),
            }, f, indent=2)
        print(f"\nSaved: {cv_path}")
        _log_elapsed("cross_validation_total", t0)

    else:
        # -------------------------------------------------------
        # Single holdout evaluation + save ensemble weights
        # -------------------------------------------------------
        test_year = args.test_year or all_years[-1]
        print(f"\nHoldout year: {test_year}")

        test_path = data_dir / f"play_states_{test_year}.csv"
        test_states = load_play_states(test_path)
        actuals = np.array([s["home_won"] for s in test_states], dtype=float)
        print(f"Test data: {len(test_states):,} plays")
        _log_elapsed("load_data", t0)

        # Train data for calibration
        train_states = []
        for year in all_years:
            if year != test_year:
                train_states.extend(
                    load_play_states(data_dir / f"play_states_{year}.csv"))
        train_actuals = np.array(
            [s["home_won"] for s in train_states], dtype=float)

        # Engine predictions
        engine_preds = {}
        engine_briers = {}

        print("\n[v1] Normal approximation + Optuna...")
        v1_preds = predict_v1(test_states)
        engine_preds["v1"] = v1_preds
        engine_briers["v1"] = compute_brier(v1_preds, actuals)
        v1_ece = compute_ece(v1_preds, actuals)
        print(f"  Brier: {engine_briers['v1']:.6f} | ECE: {v1_ece:.4f}")

        print("\n[v2] Empirical + Markov (fair holdout)...")
        v2_preds = predict_v2(test_states, data_dir, exclude_year=test_year)
        if v2_preds is not None:
            engine_preds["v2"] = v2_preds
            engine_briers["v2"] = compute_brier(v2_preds, actuals)
            v2_ece = compute_ece(v2_preds, actuals)
            print(f"  Brier: {engine_briers['v2']:.6f} | ECE: {v2_ece:.4f}")
        else:
            print("  SKIP: v2 not available")

        print("\n[bayesian] Bayesian Hierarchical...")
        bayes_preds, bayes_lower, bayes_upper = predict_bayesian(
            test_states, data_dir)
        if bayes_preds is not None:
            engine_preds["bayesian"] = bayes_preds
            engine_briers["bayesian"] = compute_brier(bayes_preds, actuals)
            bayes_ece = compute_ece(bayes_preds, actuals)
            ci_width = float(np.mean(bayes_upper - bayes_lower))
            print(f"  Brier: {engine_briers['bayesian']:.6f} | "
                  f"ECE: {bayes_ece:.4f} | Mean CI: {ci_width:.4f}")
        else:
            print("  SKIP: Bayesian not available")

        print("\n[lgbm] LightGBM (state)...")
        lgbm_preds = predict_lgbm(test_states, data_dir)
        if lgbm_preds is not None:
            engine_preds["lgbm"] = lgbm_preds
            engine_briers["lgbm"] = compute_brier(lgbm_preds, actuals)
            lgbm_ece = compute_ece(lgbm_preds, actuals)
            print(f"  Brier: {engine_briers['lgbm']:.6f} | ECE: {lgbm_ece:.4f}")
        else:
            print("  SKIP: LightGBM (state) not available")

        print("\n[statcast] LightGBM (Statcast)...")
        statcast_preds = predict_statcast(test_states, data_dir)
        if statcast_preds is not None:
            engine_preds["statcast"] = statcast_preds
            engine_briers["statcast"] = compute_brier(statcast_preds, actuals)
            statcast_ece = compute_ece(statcast_preds, actuals)
            print(f"  Brier: {engine_briers['statcast']:.6f} | ECE: {statcast_ece:.4f}")
            print("  (game-state only — live mode with pitch/hit data will be better)")
        else:
            print("  SKIP: Statcast not available")
        _log_elapsed("engine_predictions", t0)

        # Ensemble
        print(f"\n{'=' * 60}")
        print("ENSEMBLE (inverse-Brier weighted)")
        print(f"{'=' * 60}")

        weights = {n: 1.0/b for n, b in engine_briers.items()}
        total_w = sum(weights.values())
        norm_weights = {n: round(w/total_w, 4) for n, w in weights.items()}
        print(f"  Weights: {norm_weights}")

        ens_preds = ensemble_predictions(engine_preds, engine_briers)
        ens_brier = compute_brier(ens_preds, actuals)
        ens_ece = compute_ece(ens_preds, actuals)
        print(f"  Brier: {ens_brier:.6f} | ECE: {ens_ece:.4f}")

        # Calibration
        print(f"\n{'=' * 60}")
        print("CALIBRATED ENSEMBLE (Isotonic Regression)")
        print(f"{'=' * 60}")

        # Build train ensemble for calibrator training
        train_v1 = predict_v1(train_states)
        train_engine_preds = {"v1": train_v1}

        train_v2 = predict_v2(train_states, data_dir, exclude_year=test_year)
        if train_v2 is not None:
            train_engine_preds["v2"] = train_v2

        train_lgbm = predict_lgbm(train_states, data_dir)
        if train_lgbm is not None:
            train_engine_preds["lgbm"] = train_lgbm

        train_statcast = predict_statcast(train_states, data_dir)
        if train_statcast is not None:
            train_engine_preds["statcast"] = train_statcast

        train_ens = ensemble_predictions(train_engine_preds, engine_briers)

        cal_preds = calibrate_predictions(train_ens, train_actuals, ens_preds)
        cal_brier = compute_brier(cal_preds, actuals)
        cal_ece = compute_ece(cal_preds, actuals)
        print(f"  Brier: {cal_brier:.6f} | ECE: {cal_ece:.4f}")

        # Comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        best_single_name = min(engine_briers, key=engine_briers.get)
        best_single = engine_briers[best_single_name]

        print(f"  Best single engine: {best_single_name} "
              f"(Brier: {best_single:.6f})")
        print(f"  Ensemble:           Brier: {ens_brier:.6f} "
              f"({(best_single - ens_brier)/best_single*100:+.2f}% vs best single)")
        print(f"  Calibrated:         Brier: {cal_brier:.6f} "
              f"({(best_single - cal_brier)/best_single*100:+.2f}% vs best single)")

        # Save weights
        weights_data = {
            "test_year": test_year,
            "engine_briers": {n: round(b, 6) for n, b in engine_briers.items()},
            "weights": norm_weights,
            "ensemble_brier": round(ens_brier, 6),
            "ensemble_ece": round(ens_ece, 4),
            "calibrated_brier": round(cal_brier, 6),
            "calibrated_ece": round(cal_ece, 4),
        }
        weights_path = output_dir / "ensemble_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights_data, f, indent=2)
        print(f"\nSaved: {weights_path}")

        # Save calibrator
        _log_elapsed("ensemble_and_calibration", t0)

        try:
            from sklearn.isotonic import IsotonicRegression
            import joblib

            ir = IsotonicRegression(y_min=0.01, y_max=0.99,
                                    out_of_bounds="clip")
            ir.fit(train_ens, train_actuals)
            cal_path = output_dir / "calibrator.pkl"
            joblib.dump(ir, cal_path)
            print(f"Saved: {cal_path}")
        except ImportError:
            print("  sklearn/joblib not available, skipping calibrator save")


if __name__ == "__main__":
    main()
