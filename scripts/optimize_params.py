"""
Optimize WP model parameters using Optuna.

Searches for optimal values of:
- variance_factor: controls uncertainty spread in normal approximation
- scoring_factor: walk-off probability scaling (bottom 9th tie)
- behind_lambda_mult: Poisson lambda multiplier (bottom 9th, behind)
- top9_lambda_mult: Poisson lambda multiplier (top 9th, home leading)
- extras_win_prob: probability of winning in extras from tie

Minimizes Brier Score on actual play-by-play data.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna


def _log_elapsed(label: str, start: float, budget_min: int = 240):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")
from scipy.stats import norm, poisson

sys.path.insert(0, str(Path(__file__).parent.parent))
from win_probability import RE24_MLB, MLB_RPG


def _get_re24(runners: tuple, outs: int,
              runs_per_game: float = MLB_RPG) -> float:
    key = (runners[0], runners[1], runners[2], outs)
    base = RE24_MLB.get(key, 0.0)
    return base * (runs_per_game / MLB_RPG)


def calculate_wp_parametric(
    inning: int,
    top_bottom: str,
    outs: int,
    runners: tuple,
    score_diff: int,
    runs_per_game: float = MLB_RPG,
    variance_factor: float = 1.3,
    scoring_factor: float = 1.8,
    behind_lambda_mult: float = 1.5,
    top9_lambda_mult: float = 1.3,
    extras_win_prob: float = 0.50,
) -> float:
    """Parameterized WP calculation for optimization."""
    re = _get_re24(runners, outs, runs_per_game)
    rpg = runs_per_game
    runs_per_half = rpg / 18.0

    if top_bottom == "top":
        away_extra = re
        home_extra = 0.0
        away_remaining_halves = max(9 - inning, 0)
        home_remaining_halves = max(9 - inning + 1, 1)
    else:
        away_extra = 0.0
        home_extra = re
        away_remaining_halves = max(9 - inning, 0)
        home_remaining_halves = max(9 - inning, 0)

    home_mean = (
        score_diff + home_extra - away_extra
        + (home_remaining_halves - away_remaining_halves) * runs_per_half
    )
    home_std = np.sqrt(
        (home_remaining_halves + away_remaining_halves)
        * runs_per_half * variance_factor + 0.01
    )

    # Late-inning special cases
    if inning >= 9 and top_bottom == "bottom":
        if score_diff > 0:
            return 0.99
        if score_diff == 0:
            p_score_1 = 1.0 - np.exp(-re * scoring_factor)
            wp = p_score_1 + (1 - p_score_1) * extras_win_prob
            return min(0.99, max(0.01, wp))
        else:
            needed = abs(score_diff)
            lam = re * behind_lambda_mult
            p_enough = 1.0 - poisson.cdf(needed - 1, max(lam, 0.01))
            return min(0.99, max(0.01, p_enough))

    if inning >= 9 and top_bottom == "top":
        if score_diff > 0:
            lam = re * top9_lambda_mult
            p_tie_or_lead = 1.0 - poisson.cdf(
                score_diff - 1, max(lam, 0.01))
            p_lead = 1.0 - poisson.cdf(score_diff, max(lam, 0.01))
            p_home_loses = p_lead + (p_tie_or_lead - p_lead) * 0.45
            return min(0.99, max(0.01, 1.0 - p_home_loses))

    wp = norm.cdf(home_mean / max(home_std, 0.1))
    return min(0.99, max(0.01, float(wp)))


def load_play_states(csv_path: str) -> list[dict]:
    states = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    return states


def brier_score(states: list[dict], params: dict) -> float:
    """Calculate Brier score with given parameters."""
    total = 0.0
    for s in states:
        wp = calculate_wp_parametric(
            s["inning"], s["half_inning"], s["outs"],
            s["runners"], s["score_diff"],
            variance_factor=params["variance_factor"],
            scoring_factor=params["scoring_factor"],
            behind_lambda_mult=params["behind_lambda_mult"],
            top9_lambda_mult=params["top9_lambda_mult"],
            extras_win_prob=params["extras_win_prob"],
        )
        total += (wp - s["home_won"]) ** 2
    return total / len(states)


def objective(trial: optuna.Trial, states: list[dict]) -> float:
    params = {
        "variance_factor": trial.suggest_float(
            "variance_factor", 0.3, 5.0),
        "scoring_factor": trial.suggest_float(
            "scoring_factor", 0.5, 5.0),
        "behind_lambda_mult": trial.suggest_float(
            "behind_lambda_mult", 0.3, 4.0),
        "top9_lambda_mult": trial.suggest_float(
            "top9_lambda_mult", 0.3, 4.0),
        "extras_win_prob": trial.suggest_float(
            "extras_win_prob", 0.40, 0.60),
    }
    return brier_score(states, params)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimize WP model parameters with Optuna")
    parser.add_argument("--input", required=True,
                        help="Path to play_states CSV")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of Optuna trials")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to W&B")
    parser.add_argument("--wandb-project",
                        default="mlb-win-probability")
    args = parser.parse_args()

    t0 = time.time()

    output_dir = Path(args.input).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(output_dir / "optimized_params.json")

    print(f"Loading data from {args.input}")
    states = load_play_states(args.input)
    print(f"Loaded {len(states)} play states")

    # Current (default) parameters baseline
    default_params = {
        "variance_factor": 1.3,
        "scoring_factor": 1.8,
        "behind_lambda_mult": 1.5,
        "top9_lambda_mult": 1.3,
        "extras_win_prob": 0.50,
    }
    baseline_brier = brier_score(states, default_params)
    print(f"Baseline Brier Score (current params): {baseline_brier:.6f}")

    # Optuna optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    callbacks = []
    if args.wandb:
        import wandb
        from optuna.integration.wandb import WeightsAndBiasesCallback
        wandb.init(project=args.wandb_project, job_type="optimize",
                   config={"n_trials": args.n_trials,
                           "n_plays": len(states)})
        wandb.log({"baseline_brier": baseline_brier})
        callbacks.append(WeightsAndBiasesCallback(
            metric_name="brier_score", as_multirun=True))

    study = optuna.create_study(
        direction="minimize",
        study_name="wp_param_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Seed with current defaults
    study.enqueue_trial(default_params)

    _log_elapsed("load_and_baseline", t0)
    print(f"\nRunning {args.n_trials} Optuna trials...")
    study.optimize(
        lambda trial: objective(trial, states),
        n_trials=args.n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )

    best = study.best_trial
    improvement = (baseline_brier - best.value) / baseline_brier * 100

    print(f"\n{'=' * 60}")
    print("Optimization Results")
    print(f"{'=' * 60}")
    print(f"Baseline Brier: {baseline_brier:.6f}")
    print(f"Best Brier:     {best.value:.6f}")
    print(f"Improvement:    {improvement:+.2f}%")
    print(f"\nBest Parameters:")
    for k, v in sorted(best.params.items()):
        default_v = default_params[k]
        print(f"  {k}: {v:.4f} (was {default_v})")

    results = {
        "baseline_brier": baseline_brier,
        "best_brier": best.value,
        "improvement_pct": round(improvement, 2),
        "default_params": default_params,
        "best_params": best.params,
        "n_trials": args.n_trials,
        "n_play_states": len(states),
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")
    _log_elapsed("total", t0)

    if args.wandb:
        import wandb
        wandb.init(
            project="mlb-win-probability-scripts",
            name="optimization-summary",
            tags=["summary"],
        )
        wandb.log({
            "best_brier": best.value,
            "improvement_pct": improvement,
            **{f"best_{k}": v for k, v in best.params.items()},
        })
        wandb.finish()
        print("W&B logging complete")


if __name__ == "__main__":
    main()
