"""
Compare WP engines: v1 (Normal) vs v2 (Empirical+Markov) vs LightGBM.

Handles data leakage: builds v2 table EXCLUDING the test year for
fair comparison, so the empirical table never sees test outcomes.

Usage:
  python scripts/compare_wp_engines.py [--data-dir data/] [--output-dir results/]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_data(data_dir: Path,
                   test_year: int | None = None) -> tuple[list[dict], int]:
    """Load play states for the holdout test year."""
    csvs = sorted(data_dir.glob("play_states_*.csv"))
    if not csvs:
        print("ERROR: No play_states CSVs found in", data_dir)
        sys.exit(1)

    if test_year is None:
        test_year = int(csvs[-1].stem.split("_")[-1])

    csv_path = data_dir / f"play_states_{test_year}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    states = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                states.append({
                    "inning": int(row["inning"]),
                    "half_inning": row["half_inning"],
                    "outs": int(row["outs"]),
                    "runners": (int(row["runner_1b"]),
                                int(row["runner_2b"]),
                                int(row["runner_3b"])),
                    "score_diff": int(row["score_diff"]),
                    "home_won": int(row["home_won"]),
                })
            except (ValueError, KeyError):
                continue

    return states, test_year


def compute_metrics(predictions: np.ndarray,
                    actuals: np.ndarray) -> dict:
    """Compute Brier, BSS, Log Loss, ECE, and calibration."""
    brier = float(np.mean((predictions - actuals) ** 2))
    brier_baseline = float(np.mean((0.5 - actuals) ** 2))
    brier_skill = 1 - brier / brier_baseline

    eps = 1e-7
    p_clip = np.clip(predictions, eps, 1 - eps)
    log_loss = float(-np.mean(
        actuals * np.log(p_clip) + (1 - actuals) * np.log(1 - p_clip)
    ))

    calibration = []
    ece = 0.0
    n_total = len(actuals)
    for low in np.arange(0, 1.0, 0.1):
        high = low + 0.1
        mask = (predictions >= low) & (predictions < high)
        if mask.sum() > 0:
            pred_mean = float(predictions[mask].mean())
            act_mean = float(actuals[mask].mean())
            abs_err = abs(pred_mean - act_mean)
            ece += abs_err * mask.sum() / n_total
            calibration.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "predicted": round(pred_mean, 4),
                "actual": round(act_mean, 4),
                "error": round(abs_err, 4),
                "n": int(mask.sum()),
            })

    # MAE by inning
    innings = np.array([0] * len(actuals))  # placeholder
    mae_by_inning = {}

    return {
        "brier": round(brier, 6),
        "brier_skill": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
        "calibration": calibration,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare WP engines (v1 vs v2 vs LightGBM)")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--test-year", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    states, test_year = load_test_data(data_dir, args.test_year)
    n_games = len(set())  # can't compute without game_pk
    print(f"Test data: {len(states):,} plays from {test_year}")
    actuals = np.array([s["home_won"] for s in states], dtype=float)

    results = {}

    # -------------------------------------------------------
    # Engine A: v1 (Normal approximation, current production)
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[A] v1 — Normal approximation + Optuna params")
    print(f"{'=' * 60}")
    from win_probability import calculate_wp, MLB_RPG

    v1_preds = np.array([
        calculate_wp(s["inning"], s["half_inning"], s["outs"],
                     s["runners"], s["score_diff"], MLB_RPG)
        for s in states
    ])
    results["v1_normal"] = compute_metrics(v1_preds, actuals)
    print(f"  Brier: {results['v1_normal']['brier']:.6f} | "
          f"BSS: {results['v1_normal']['brier_skill']:.4f} | "
          f"ECE: {results['v1_normal']['ece']:.4f}")

    # -------------------------------------------------------
    # Engine B: v2 (Empirical + Markov)
    # Build table EXCLUDING test year to prevent data leakage
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[B] v2 — Empirical WP table + Markov chain "
          f"(excluding {test_year})")
    print(f"{'=' * 60}")

    from scripts.build_wp_v2 import (
        load_all_play_states,
        extract_transitions,
        build_transition_matrix,
        simulate_run_distributions,
        build_empirical_wp,
        compute_markov_wp,
    )

    try:
        print("  Building fair-comparison tables (excluding test year)...")
        train_states = load_all_play_states(data_dir, {test_year})

        if train_states:
            transitions = extract_transitions(train_states)
            trans_probs = build_transition_matrix(transitions)
            run_dists = simulate_run_distributions(trans_probs, n_sims=200_000)
            wp_table = build_empirical_wp(train_states, run_dists)

            # Use the fair table for predictions
            from win_probability_v2 import WPEngineV2
            engine_v2 = WPEngineV2.__new__(WPEngineV2)
            engine_v2.wp_table = wp_table
            engine_v2.run_dists = run_dists
            engine_v2._loaded = True

            v2_preds = np.array([
                engine_v2.calculate_wp(
                    s["inning"], s["half_inning"], s["outs"],
                    s["runners"], s["score_diff"], MLB_RPG)
                for s in states
            ])
            results["v2_empirical_markov"] = compute_metrics(v2_preds, actuals)
            print(f"  Brier: {results['v2_empirical_markov']['brier']:.6f} | "
                  f"BSS: {results['v2_empirical_markov']['brier_skill']:.4f} | "
                  f"ECE: {results['v2_empirical_markov']['ece']:.4f}")
        else:
            print("  SKIP: No training data available")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------
    # Engine C: LightGBM (already trained with holdout)
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[C] LightGBM ML model")
    print(f"{'=' * 60}")

    lgbm_path = data_dir / "wp_lgbm_model.txt"
    if lgbm_path.exists():
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(lgbm_path))

            X_test = np.array([
                [
                    s["inning"],
                    1 if s["half_inning"] == "bottom" else 0,
                    s["outs"],
                    s["runners"][0], s["runners"][1], s["runners"][2],
                    s["score_diff"],
                    s["inning"] * (1 if s["half_inning"] == "bottom" else 0),
                    abs(s["score_diff"]),
                    sum(s["runners"]),
                    max(9 - s["inning"], 0) * (
                        1 if s["score_diff"] == 0 else 0),
                ]
                for s in states
            ])
            lgbm_preds = model.predict(X_test)
            results["lgbm"] = compute_metrics(lgbm_preds, actuals)
            print(f"  Brier: {results['lgbm']['brier']:.6f} | "
                  f"BSS: {results['lgbm']['brier_skill']:.4f} | "
                  f"ECE: {results['lgbm']['ece']:.4f}")
        except ImportError:
            print("  SKIP: lightgbm not installed")
    else:
        print("  SKIP: model file not found")

    # -------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"COMPARISON SUMMARY — Test Year: {test_year} "
          f"({len(states):,} plays)")
    print(f"{'=' * 70}")
    print(f"{'Engine':<25} {'Brier':>10} {'BSS':>8} "
          f"{'Log Loss':>10} {'ECE':>8}")
    print(f"{'-' * 70}")

    for name, m in results.items():
        print(f"{name:<25} {m['brier']:>10.6f} {m['brier_skill']:>8.4f} "
              f"{m['log_loss']:>10.6f} {m['ece']:>8.4f}")

    # Improvements over v1
    if "v1_normal" in results:
        v1_brier = results["v1_normal"]["brier"]
        print()
        for name, m in results.items():
            if name != "v1_normal":
                imp = (v1_brier - m["brier"]) / v1_brier * 100
                print(f"  {name} vs v1: {imp:+.2f}% Brier improvement")

    # Calibration table
    print(f"\n{'=' * 70}")
    print("CALIBRATION COMPARISON (|predicted - actual| per decile)")
    print(f"{'=' * 70}")
    header = f"{'Bin':<12}"
    for name in results:
        header += f" {name:>20}"
    print(header)
    print("-" * (12 + 20 * len(results) + len(results)))

    all_bins = sorted({c["bin"] for m in results.values()
                       for c in m["calibration"]})
    for bin_name in all_bins:
        row = f"{bin_name:<12}"
        for name in results:
            cal_dict = {c["bin"]: c for c in results[name]["calibration"]}
            if bin_name in cal_dict:
                row += f" {cal_dict[bin_name]['error']:>20.4f}"
            else:
                row += f" {'---':>20}"
        print(row)

    # Save results
    comparison = {
        "test_year": test_year,
        "n_plays": len(states),
        "results": results,
    }
    comparison_path = output_dir / "wp_engine_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {comparison_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Calibration plot
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1,
                label="Perfect")
        colors = {
            "v1_normal": "#e65100",
            "v2_empirical_markov": "#1976d2",
            "lgbm": "#388e3c",
        }
        for name, m in results.items():
            preds_cal = [c["predicted"] for c in m["calibration"]]
            acts_cal = [c["actual"] for c in m["calibration"]]
            color = colors.get(name, "#777")
            ax.plot(preds_cal, acts_cal, "o-", color=color,
                    label=f"{name} (Brier={m['brier']:.4f})",
                    markersize=6, linewidth=2)
        ax.set_xlabel("Predicted WP", fontsize=14)
        ax.set_ylabel("Actual Win Rate", fontsize=14)
        ax.set_title("Calibration — Engine Comparison", fontsize=16)
        ax.legend(fontsize=11, loc="upper left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

        # Brier score bar chart
        ax = axes[1]
        names = list(results.keys())
        briers = [results[n]["brier"] for n in names]
        bar_colors = [colors.get(n, "#777") for n in names]
        bars = ax.bar(names, briers, color=bar_colors, alpha=0.8)
        ax.set_ylabel("Brier Score (lower = better)", fontsize=14)
        ax.set_title("Brier Score Comparison", fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, briers):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", fontsize=12,
                    fontweight="bold")

        plt.tight_layout()
        plot_path = output_dir / "wp_engine_comparison.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
