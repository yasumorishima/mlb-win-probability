"""
Validate WP model against actual MLB game outcomes.

Metrics:
- Brier Score: mean((predicted - actual)^2), lower is better
- Brier Skill Score: 1 - brier/brier_baseline (>0 means better than 50/50)
- Log Loss: proper scoring rule for probabilities
- Calibration: predicted WP bins vs actual win rate
- MAE by inning, score differential, base-out state
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from win_probability import calculate_wp, MLB_RPG


def load_play_states(csv_path: str) -> list[dict]:
    states = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append({
                "game_pk": int(row["game_pk"]),
                "inning": int(row["inning"]),
                "half_inning": row["half_inning"],
                "outs": int(row["outs"]),
                "runners": (int(row["runner_1b"]),
                            int(row["runner_2b"]),
                            int(row["runner_3b"])),
                "score_diff": int(row["score_diff"]),
                "home_won": int(row["home_won"]),
            })
    return states


def compute_metrics(states: list[dict]) -> dict:
    """Compute all validation metrics."""
    predictions = []
    actuals = []
    innings = []
    score_diffs = []
    base_out_states = []

    for s in states:
        wp = calculate_wp(
            s["inning"], s["half_inning"], s["outs"],
            s["runners"], s["score_diff"], MLB_RPG,
        )
        predictions.append(wp)
        actuals.append(s["home_won"])
        innings.append(s["inning"])
        score_diffs.append(s["score_diff"])
        base_out_states.append(
            (s["runners"][0], s["runners"][1], s["runners"][2], s["outs"])
        )

    preds = np.array(predictions)
    acts = np.array(actuals, dtype=float)
    inns = np.array(innings)
    diffs = np.array(score_diffs)

    # --- Brier Score ---
    brier = float(np.mean((preds - acts) ** 2))
    brier_baseline = float(np.mean((0.5 - acts) ** 2))
    brier_skill = 1 - brier / brier_baseline

    # --- Log Loss ---
    eps = 1e-7
    p_clip = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(
        acts * np.log(p_clip) + (1 - acts) * np.log(1 - p_clip)
    ))

    # --- Calibration by decile ---
    calibration = []
    for low in np.arange(0, 1.0, 0.1):
        high = low + 0.1
        mask = (preds >= low) & (preds < high)
        if mask.sum() > 0:
            calibration.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "mean_predicted": round(float(preds[mask].mean()), 4),
                "actual_win_rate": round(float(acts[mask].mean()), 4),
                "count": int(mask.sum()),
                "abs_error": round(
                    abs(float(preds[mask].mean()) - float(acts[mask].mean())), 4
                ),
            })

    # --- Calibration error (ECE: Expected Calibration Error) ---
    total = len(states)
    ece = sum(c["abs_error"] * c["count"] / total for c in calibration)

    # --- MAE by inning ---
    mae_by_inning = {}
    for inn in range(1, 13):
        mask = inns == inn
        if mask.sum() > 0:
            mae = float(np.mean(np.abs(preds[mask] - acts[mask])))
            mae_by_inning[str(inn)] = {
                "mae": round(mae, 4), "count": int(mask.sum()),
            }
    mask = inns >= 13
    if mask.sum() > 0:
        mae_by_inning["13+"] = {
            "mae": round(float(np.mean(np.abs(preds[mask] - acts[mask]))), 4),
            "count": int(mask.sum()),
        }

    # --- MAE by score differential ---
    mae_by_diff = {}
    for d in range(-8, 9):
        mask = diffs == d
        if mask.sum() >= 50:
            mae_by_diff[str(d)] = {
                "mae": round(
                    float(np.mean(np.abs(preds[mask] - acts[mask]))), 4
                ),
                "count": int(mask.sum()),
            }

    # --- MAE by base-out state (24 states) ---
    runner_labels = {
        (0, 0, 0): "---", (1, 0, 0): "1--", (0, 1, 0): "-2-",
        (1, 1, 0): "12-", (0, 0, 1): "--3", (1, 0, 1): "1-3",
        (0, 1, 1): "-23", (1, 1, 1): "123",
    }
    mae_by_bo = {}
    for bo_state, runs_list in defaultdict(list).items():
        pass  # placeholder
    # Compute properly
    bo_groups = defaultdict(lambda: {"preds": [], "acts": []})
    for i, bo in enumerate(base_out_states):
        bo_groups[bo]["preds"].append(preds[i])
        bo_groups[bo]["acts"].append(acts[i])
    for bo, data in sorted(bo_groups.items()):
        r = (bo[0], bo[1], bo[2])
        label = f"{runner_labels.get(r, '???')}/{bo[3]}out"
        p = np.array(data["preds"])
        a = np.array(data["acts"])
        mae_by_bo[label] = {
            "mae": round(float(np.mean(np.abs(p - a))), 4),
            "count": len(data["preds"]),
        }

    # --- Late inning accuracy (7th+) ---
    late_mask = inns >= 7
    if late_mask.sum() > 0:
        late_brier = float(np.mean((preds[late_mask] - acts[late_mask]) ** 2))
    else:
        late_brier = None

    return {
        "n_plays": len(states),
        "n_games": len(set(s["game_pk"] for s in states)),
        "brier_score": round(brier, 6),
        "brier_baseline": round(brier_baseline, 6),
        "brier_skill_score": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
        "late_inning_brier": round(late_brier, 6) if late_brier else None,
        "calibration": calibration,
        "mae_by_inning": mae_by_inning,
        "mae_by_score_diff": mae_by_diff,
        "mae_by_base_out": mae_by_bo,
    }


def plot_calibration(calibration: list[dict], output_path: str):
    """Generate calibration + distribution plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = [c["mean_predicted"] for c in calibration]
    actual = [c["actual_win_rate"] for c in calibration]
    counts = [c["count"] for c in calibration]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1,
             label="Perfect calibration")
    ax1.scatter(predicted, actual,
                s=[max(30, c / 50) for c in counts],
                alpha=0.8, color="#1976d2", zorder=5)
    ax1.plot(predicted, actual, "-", color="#1976d2", alpha=0.6)
    ax1.set_xlabel("Predicted Win Probability", fontsize=14)
    ax1.set_ylabel("Actual Win Rate", fontsize=14)
    ax1.set_title("Calibration Curve", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Distribution histogram
    bins = [c["bin"] for c in calibration]
    ax2.bar(bins, counts, color="#1976d2", alpha=0.7)
    ax2.set_xlabel("Predicted WP Bin", fontsize=14)
    ax2.set_ylabel("Number of Play States", fontsize=14)
    ax2.set_title("Distribution of Predictions", fontsize=16)
    ax2.tick_params(axis="x", rotation=45, labelsize=11)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {output_path}")


def plot_mae_by_inning(mae_by_inning: dict, output_path: str):
    """MAE breakdown by inning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    innings = list(mae_by_inning.keys())
    maes = [v["mae"] for v in mae_by_inning.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(innings, maes, color="#e65100", alpha=0.8)
    ax.set_xlabel("Inning", fontsize=14)
    ax.set_ylabel("Mean Absolute Error", fontsize=14)
    ax.set_title("WP Model Error by Inning", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MAE by inning plot saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate WP model against real MLB outcomes")
    parser.add_argument("--input", required=True,
                        help="Path to play_states CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files")
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to W&B")
    parser.add_argument("--wandb-project",
                        default="mlb-win-probability")
    args = parser.parse_args()

    output_dir = (Path(args.output_dir)
                  if args.output_dir
                  else Path(args.input).parent.parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}")
    states = load_play_states(args.input)
    n_games = len(set(s["game_pk"] for s in states))
    print(f"Loaded {len(states)} play states from {n_games} games")

    print("Computing metrics...")
    metrics = compute_metrics(states)

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print("WP Model Validation Results")
    print(f"{'=' * 60}")
    print(f"Games: {metrics['n_games']}")
    print(f"Play States: {metrics['n_plays']}")
    print(f"Brier Score: {metrics['brier_score']:.6f} "
          f"(baseline 0.5: {metrics['brier_baseline']:.6f})")
    print(f"Brier Skill Score: {metrics['brier_skill_score']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    print(f"ECE (Expected Calibration Error): {metrics['ece']:.4f}")
    if metrics["late_inning_brier"]:
        print(f"Late Inning Brier (7th+): {metrics['late_inning_brier']:.6f}")

    print(f"\nCalibration:")
    for c in metrics["calibration"]:
        diff = c["actual_win_rate"] - c["mean_predicted"]
        arrow = "^" if diff > 0.02 else "v" if diff < -0.02 else "="
        print(f"  {c['bin']}: pred={c['mean_predicted']:.3f} "
              f"actual={c['actual_win_rate']:.3f} "
              f"{arrow} (n={c['count']})")

    print(f"\nMAE by Inning:")
    for inn, v in sorted(metrics["mae_by_inning"].items(),
                         key=lambda x: (len(x[0]), x[0])):
        print(f"  Inning {inn}: MAE={v['mae']:.4f} (n={v['count']})")

    # --- Save outputs ---
    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    plot_calibration(
        metrics["calibration"],
        str(output_dir / "calibration_plot.png"),
    )
    plot_mae_by_inning(
        metrics["mae_by_inning"],
        str(output_dir / "mae_by_inning.png"),
    )

    # --- W&B ---
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, job_type="validation")
        wandb.log({
            "brier_score": metrics["brier_score"],
            "brier_skill_score": metrics["brier_skill_score"],
            "log_loss": metrics["log_loss"],
            "ece": metrics["ece"],
            "late_inning_brier": metrics["late_inning_brier"],
            "n_games": metrics["n_games"],
            "n_plays": metrics["n_plays"],
        })
        cal_table = wandb.Table(
            columns=["bin", "predicted", "actual", "count", "abs_error"],
            data=[
                [c["bin"], c["mean_predicted"], c["actual_win_rate"],
                 c["count"], c["abs_error"]]
                for c in metrics["calibration"]
            ],
        )
        wandb.log({"calibration_table": cal_table})
        wandb.log({
            "calibration_plot": wandb.Image(
                str(output_dir / "calibration_plot.png")),
            "mae_by_inning_plot": wandb.Image(
                str(output_dir / "mae_by_inning.png")),
        })
        for inn, v in metrics["mae_by_inning"].items():
            wandb.log({f"mae_inning_{inn}": v["mae"]})
        wandb.finish()
        print("W&B logging complete")


if __name__ == "__main__":
    main()
