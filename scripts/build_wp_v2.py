"""
Build empirical WP table + Markov chain from historical play-by-play data.

Phase A: Empirical WP table — state -> P(home wins), Bayesian smoothed
Phase B: Markov chain — transition matrix + half-inning run distributions

Reads: data/play_states_{year}.csv (multiple years)
Outputs:
  - data/empirical_wp_table.json  — (state -> smoothed WP) lookup
  - data/markov_transitions.json  — 24-state transition probabilities
  - data/halfinn_run_dist.json    — P(score k runs | starting state)

Usage:
  python scripts/build_wp_v2.py [--data-dir data/] [--output-dir data/]
  python scripts/build_wp_v2.py --exclude-years 2024  # for fair holdout comparison
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_RUNS = 15       # cap run distributions at 15+
PRIOR_STRENGTH = 20 # Bayesian smoothing: equivalent to 20 pseudo-observations
EXTRAS_HOME_WIN = 0.52  # home team win probability in extras
N_SIMS = 500_000    # half-inning simulations per starting state


# ============================================================
# Data Loading
# ============================================================

def load_all_play_states(data_dir: Path,
                         exclude_years: set[int] | None = None) -> list[dict]:
    """Load play states from all available year CSVs."""
    exclude_years = exclude_years or set()
    all_states = []

    for csv_path in sorted(data_dir.glob("play_states_*.csv")):
        year = int(csv_path.stem.split("_")[-1])
        if year in exclude_years:
            print(f"  Skipping {csv_path.name} (excluded)")
            continue

        n = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    all_states.append({
                        "game_pk": int(row["game_pk"]),
                        "year": year,
                        "play_idx": int(row.get("play_idx", 0)),
                        "inning": int(row["inning"]),
                        "half_inning": row["half_inning"],
                        "outs": int(row["outs"]),
                        "r1": int(row["runner_1b"]),
                        "r2": int(row["runner_2b"]),
                        "r3": int(row["runner_3b"]),
                        "home_score": int(row["home_score"]),
                        "away_score": int(row["away_score"]),
                        "score_diff": int(row["score_diff"]),
                        "home_won": int(row["home_won"]),
                        "home_score_after": int(row["home_score_after"]),
                        "away_score_after": int(row["away_score_after"]),
                    })
                    n += 1
                except (ValueError, KeyError) as e:
                    continue
        print(f"  {csv_path.name}: {n:,} plays")

    return all_states


# ============================================================
# Phase B: Markov Chain Transitions
# ============================================================

def _bo_key(outs: int, r1: int, r2: int, r3: int) -> str:
    """Base-out state key: '0_000', '1_110', '3_000' (absorbing)."""
    return f"{outs}_{r1}{r2}{r3}"


def extract_transitions(states: list[dict]) -> list[tuple[str, str, int]]:
    """Extract (state_before, state_after, runs_scored) from play-by-play.

    Groups plays by (game_pk, inning, half_inning) and tracks consecutive
    pre-play states to infer transitions.
    """
    # Group by half-inning
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in states:
        key = (s["game_pk"], s["inning"], s["half_inning"])
        groups[key].append(s)

    transitions = []
    for _, plays in groups.items():
        plays.sort(key=lambda x: x["play_idx"])

        for i in range(len(plays)):
            p = plays[i]
            before = _bo_key(p["outs"], p["r1"], p["r2"], p["r3"])

            # Runs scored by batting team during this plate appearance
            if p["half_inning"] == "bottom":
                runs = p["home_score_after"] - p["home_score"]
            else:
                runs = p["away_score_after"] - p["away_score"]
            runs = max(0, runs)

            if i + 1 < len(plays):
                nxt = plays[i + 1]
                after = _bo_key(nxt["outs"], nxt["r1"], nxt["r2"], nxt["r3"])
            else:
                # Half-inning ended (3rd out)
                after = "3_000"

            transitions.append((before, after, min(runs, MAX_RUNS)))

    return transitions


def build_transition_matrix(transitions: list[tuple[str, str, int]]) -> dict:
    """Build empirical transition probabilities from observed transitions.

    Returns dict: from_key -> list of {to, runs, count, prob}
    """
    counts: dict[str, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))

    for before, after, runs in transitions:
        counts[before][(after, runs)] += 1

    trans_probs = {}
    for from_key, destinations in counts.items():
        total = sum(destinations.values())
        entries = []
        for (to_key, runs), count in sorted(destinations.items(),
                                             key=lambda x: -x[1]):
            entries.append({
                "to": to_key,
                "runs": runs,
                "count": count,
                "prob": count / total,
            })
        trans_probs[from_key] = entries

    return trans_probs


# ============================================================
# Phase B: Half-Inning Run Distributions (via Monte Carlo)
# ============================================================

def simulate_run_distributions(trans_probs: dict,
                                n_sims: int = N_SIMS) -> dict[str, np.ndarray]:
    """Simulate half-innings from each starting state to get run distributions.

    Returns: state_key -> numpy array P(score k runs) for k=0..MAX_RUNS
    """
    # Build fast lookup for simulation
    fast_lookup: dict[str, tuple[np.ndarray, list[str], list[int]]] = {}
    for from_key, entries in trans_probs.items():
        if from_key.startswith("3_"):
            continue  # absorbing state
        probs = np.array([e["prob"] for e in entries])
        cum_probs = np.cumsum(probs)
        to_keys = [e["to"] for e in entries]
        runs_list = [e["runs"] for e in entries]
        fast_lookup[from_key] = (cum_probs, to_keys, runs_list)

    # All 24 active base-out states
    all_start_states = []
    for outs in range(3):
        for r1 in range(2):
            for r2 in range(2):
                for r3 in range(2):
                    all_start_states.append(_bo_key(outs, r1, r2, r3))

    run_dists: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    for start_key in all_start_states:
        if start_key not in fast_lookup:
            # No empirical data — use generic fallback
            dist = np.zeros(MAX_RUNS + 1)
            outs_val = int(start_key[0])
            # Higher outs = less expected scoring
            if outs_val == 0:
                dist[:6] = [0.70, 0.15, 0.07, 0.04, 0.02, 0.02]
            elif outs_val == 1:
                dist[:5] = [0.80, 0.12, 0.05, 0.02, 0.01]
            else:
                dist[:4] = [0.90, 0.07, 0.02, 0.01]
            run_dists[start_key] = dist
            continue

        runs_count = np.zeros(MAX_RUNS + 1)

        for _ in range(n_sims):
            current = start_key
            total_runs = 0
            steps = 0

            while not current.startswith("3_") and steps < 50:
                if current not in fast_lookup:
                    break
                cum_probs, to_keys, runs_arr = fast_lookup[current]
                idx = int(np.searchsorted(cum_probs, rng.random()))
                idx = min(idx, len(to_keys) - 1)
                total_runs += runs_arr[idx]
                current = to_keys[idx]
                steps += 1

            runs_count[min(total_runs, MAX_RUNS)] += 1

        run_dists[start_key] = runs_count / n_sims

    return run_dists


# ============================================================
# Markov-based WP Computation
# ============================================================

def _convolve_dists(dists: list[np.ndarray],
                    max_len: int = MAX_RUNS + 1) -> np.ndarray:
    """Convolve multiple discrete run distributions."""
    if not dists:
        result = np.zeros(max_len)
        result[0] = 1.0
        return result

    result = dists[0].copy()
    for d in dists[1:]:
        result = np.convolve(result, d)[:max_len]

    s = result.sum()
    if s > 0:
        result /= s
    return result


def compute_markov_wp(inning: int, top_bottom: str,
                      outs: int, r1: int, r2: int, r3: int,
                      score_diff: int,
                      run_dists: dict[str, np.ndarray]) -> float:
    """Compute WP by convolving remaining half-inning run distributions."""
    current_key = _bo_key(outs, r1, r2, r3)
    fresh_key = "0_000"

    if current_key not in run_dists:
        current_key = fresh_key

    max_len = MAX_RUNS + 1
    current_dist = run_dists[current_key][:max_len]
    fresh_dist = run_dists[fresh_key][:max_len]

    if top_bottom == "top":
        # Away team batting: current half + future tops
        away_future = max(9 - inning, 0)
        home_future = max(9 - inning + 1, 1)

        away_dists = [current_dist] + [fresh_dist] * away_future
        home_dists = [fresh_dist] * home_future
    else:
        # Home team batting: current half + future bottoms
        home_future = max(9 - inning, 0)
        away_future = max(9 - inning, 0)

        home_dists = [current_dist] + [fresh_dist] * home_future
        away_dists = [fresh_dist] * away_future if away_future > 0 else []

    home_total = _convolve_dists(home_dists, max_len)
    away_total = _convolve_dists(away_dists, max_len)

    # WP = P(home_remaining + score_diff > away_remaining)
    #    + EXTRAS_HOME_WIN * P(tie after 9 innings)
    wp = 0.0
    for h in range(max_len):
        if home_total[h] < 1e-10:
            continue
        for a in range(max_len):
            if away_total[a] < 1e-10:
                continue
            diff = score_diff + h - a
            if diff > 0:
                wp += home_total[h] * away_total[a]
            elif diff == 0:
                wp += home_total[h] * away_total[a] * EXTRAS_HOME_WIN

    return max(0.01, min(0.99, wp))


# ============================================================
# Phase A: Empirical WP Table (Bayesian smoothed)
# ============================================================

def _wp_state_key(inning: int, half: str,
                  outs: int, r1: int, r2: int, r3: int,
                  score_diff: int) -> str:
    """Key for the WP lookup table. Caps inning at 10+, score_diff at +/-10."""
    inn_cap = min(inning, 10)
    diff_cap = max(-10, min(10, score_diff))
    return f"{inn_cap}_{half}_{outs}_{r1}{r2}{r3}_{diff_cap}"


def build_empirical_wp(states: list[dict],
                       run_dists: dict[str, np.ndarray]) -> dict:
    """Build Bayesian-smoothed empirical WP table.

    Prior: Markov-derived WP (informative)
    Likelihood: empirical win counts
    Posterior: Beta(alpha0 + wins, beta0 + losses)
    """
    # Aggregate wins/totals per state
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})

    for s in states:
        key = _wp_state_key(s["inning"], s["half_inning"],
                            s["outs"], s["r1"], s["r2"], s["r3"],
                            s["score_diff"])
        counts[key]["wins"] += s["home_won"]
        counts[key]["total"] += 1

    wp_table: dict[str, dict] = {}

    for key, data in counts.items():
        # Parse key to compute Markov prior
        parts = key.split("_")
        inn = int(parts[0])
        half = parts[1]
        outs_val = int(parts[2])
        runners = parts[3]
        r1_v, r2_v, r3_v = int(runners[0]), int(runners[1]), int(runners[2])
        diff = int(parts[4])

        # Markov prior
        prior_wp = compute_markov_wp(inn, half, outs_val, r1_v, r2_v, r3_v,
                                     diff, run_dists)

        # Bayesian smoothing: Beta(alpha0 + wins, beta0 + losses)
        alpha0 = PRIOR_STRENGTH * prior_wp
        beta0 = PRIOR_STRENGTH * (1 - prior_wp)

        wins = data["wins"]
        total = data["total"]

        smoothed_wp = (wins + alpha0) / (total + PRIOR_STRENGTH)

        wp_table[key] = {
            "wp": round(float(smoothed_wp), 6),
            "empirical_wp": round(wins / total, 6) if total > 0 else None,
            "markov_wp": round(float(prior_wp), 6),
            "n": total,
        }

    return wp_table


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build empirical WP table + Markov chain from play-by-play data")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory containing play_states CSVs")
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for JSON files")
    parser.add_argument("--exclude-years", default="",
                        help="Comma-separated years to exclude (e.g. '2024')")
    parser.add_argument("--n-sims", type=int, default=N_SIMS,
                        help=f"Simulations per starting state (default: {N_SIMS})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude = set()
    if args.exclude_years:
        exclude = {int(y.strip()) for y in args.exclude_years.split(",")}

    # 1. Load data
    print("=" * 60)
    print("Phase 0: Loading play-by-play data")
    print("=" * 60)
    states = load_all_play_states(data_dir, exclude)
    if not states:
        print("ERROR: No play states found. Run fetch_game_states.py first.")
        sys.exit(1)

    years = sorted(set(s["year"] for s in states))
    n_games = len(set(s["game_pk"] for s in states))
    print(f"  Total: {len(states):,} plays from {n_games:,} games "
          f"({years[0]}-{years[-1]})")
    if exclude:
        print(f"  Excluded years: {sorted(exclude)}")

    # 2. Extract transitions
    print(f"\n{'=' * 60}")
    print("Phase B-1: Extracting Markov chain transitions")
    print(f"{'=' * 60}")
    transitions = extract_transitions(states)
    print(f"  {len(transitions):,} transitions extracted")

    # 3. Build transition matrix
    print(f"\n{'=' * 60}")
    print("Phase B-2: Building transition probability matrix")
    print(f"{'=' * 60}")
    trans_probs = build_transition_matrix(transitions)
    n_active = sum(1 for k in trans_probs if not k.startswith("3_"))
    print(f"  {n_active} active states with transition data")

    # Show top transitions from bases-empty 0-out
    if "0_000" in trans_probs:
        top5 = trans_probs["0_000"][:5]
        print(f"  Top transitions from 0_000 (bases empty, 0 out):")
        for t in top5:
            print(f"    -> {t['to']} (p={t['prob']:.3f}, runs={t['runs']}, "
                  f"n={t['count']:,})")

    # 4. Simulate run distributions
    print(f"\n{'=' * 60}")
    print(f"Phase B-3: Simulating half-inning run distributions "
          f"({args.n_sims:,} sims)")
    print(f"{'=' * 60}")
    run_dists = simulate_run_distributions(trans_probs, args.n_sims)
    print(f"  {len(run_dists)} starting states computed")

    # Sanity check: expected runs from bases-empty start
    fresh = run_dists.get("0_000", np.zeros(MAX_RUNS + 1))
    expected = sum(k * fresh[k] for k in range(len(fresh)))
    print(f"  Expected runs from 0_000: {expected:.3f} "
          f"(MLB avg per half-inning ~0.50)")
    print(f"  P(0 runs from 0_000): {fresh[0]:.3f}")
    print(f"  P(1+ runs from 0_000): {1 - fresh[0]:.3f}")

    # 5. Build empirical WP table
    print(f"\n{'=' * 60}")
    print("Phase A: Building Bayesian-smoothed empirical WP table")
    print(f"{'=' * 60}")
    wp_table = build_empirical_wp(states, run_dists)
    print(f"  {len(wp_table)} states in WP table")

    # Coverage analysis
    n_thin = sum(1 for v in wp_table.values() if v["n"] < 20)
    n_medium = sum(1 for v in wp_table.values() if 20 <= v["n"] < 100)
    n_rich = sum(1 for v in wp_table.values() if v["n"] >= 100)
    print(f"  Coverage: <20 obs = {n_thin} | 20-99 = {n_medium} | 100+ = {n_rich}")

    # Spot checks
    for check_key, desc in [
        ("1_top_0_000_0", "1T 0out empty tied"),
        ("5_bottom_1_100_-1", "5B 1out 1B, down 1"),
        ("9_bottom_2_111_0", "9B 2out loaded, tied"),
    ]:
        if check_key in wp_table:
            entry = wp_table[check_key]
            print(f"  {desc}: wp={entry['wp']:.3f} "
                  f"(emp={entry['empirical_wp']:.3f}, "
                  f"markov={entry['markov_wp']:.3f}, n={entry['n']})")

    # 6. Save outputs
    print(f"\n{'=' * 60}")
    print("Saving outputs")
    print(f"{'=' * 60}")

    trans_path = output_dir / "markov_transitions.json"
    with open(trans_path, "w") as f:
        json.dump(trans_probs, f)
    print(f"  {trans_path} ({trans_path.stat().st_size / 1024:.0f} KB)")

    rd_path = output_dir / "halfinn_run_dist.json"
    rd_serializable = {k: v.tolist() for k, v in run_dists.items()}
    with open(rd_path, "w") as f:
        json.dump(rd_serializable, f)
    print(f"  {rd_path} ({rd_path.stat().st_size / 1024:.0f} KB)")

    wp_path = output_dir / "empirical_wp_table.json"
    with open(wp_path, "w") as f:
        json.dump(wp_table, f)
    print(f"  {wp_path} ({wp_path.stat().st_size / 1024:.0f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
