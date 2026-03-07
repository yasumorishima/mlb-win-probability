"""
Recalculate RE24 table from actual MLB play-by-play data.

For each half-inning:
1. Track the base-out state at each PA
2. Calculate total runs scored by the batting team in that half-inning
3. For each PA, record "runs from this point to end of half-inning"
4. Aggregate by 24 base-out states -> new RE24 table

Compares computed values against hardcoded RE24_MLB (2010-2019 averages).
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from win_probability import RE24_MLB


RUNNER_LABELS = {
    (0, 0, 0): "---", (1, 0, 0): "1--", (0, 1, 0): "-2-",
    (1, 1, 0): "12-", (0, 0, 1): "--3", (1, 0, 1): "1-3",
    (0, 1, 1): "-23", (1, 1, 1): "123",
}


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
                "runner_1b": int(row["runner_1b"]),
                "runner_2b": int(row["runner_2b"]),
                "runner_3b": int(row["runner_3b"]),
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "home_score_after": int(row["home_score_after"]),
                "away_score_after": int(row["away_score_after"]),
            })
    return states


def compute_re24(states: list[dict]) -> dict:
    """
    Compute RE24 from play-by-play data.

    Groups plays by half-inning, then for each PA records:
    - The base-out state at that PA
    - Runs scored by the batting team from that PA to end of half-inning
    """
    # Group plays by half-inning
    half_innings: dict[tuple, list[dict]] = defaultdict(list)
    for s in states:
        key = (s["game_pk"], s["inning"], s["half_inning"])
        half_innings[key].append(s)

    # Collect runs-to-end for each base-out state
    re24_samples: dict[tuple, list[float]] = defaultdict(list)

    for _hi_key, plays in half_innings.items():
        half = plays[0]["half_inning"]

        # Determine batting team's score at start and end of half-inning
        if half == "top":
            hi_start = plays[0]["away_score"]
            hi_end = plays[-1]["away_score_after"]
        else:
            hi_start = plays[0]["home_score"]
            hi_end = plays[-1]["home_score_after"]

        total_runs = hi_end - hi_start

        for play in plays:
            if half == "top":
                runs_so_far = play["away_score"] - hi_start
            else:
                runs_so_far = play["home_score"] - hi_start

            runs_to_end = total_runs - runs_so_far

            state_key = (
                play["runner_1b"], play["runner_2b"],
                play["runner_3b"], play["outs"],
            )
            re24_samples[state_key].append(runs_to_end)

    # Compute statistics for each state
    result = {}
    for state_key in sorted(re24_samples.keys()):
        samples = re24_samples[state_key]
        arr = np.array(samples)
        hardcoded = RE24_MLB.get(state_key)

        runners = (state_key[0], state_key[1], state_key[2])
        outs = state_key[3]
        label = f"{RUNNER_LABELS.get(runners, '???')} {outs}out"

        result[label] = {
            "state": list(state_key),
            "computed_re": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "median": round(float(np.median(arr)), 3),
            "count": len(samples),
            "hardcoded_re": hardcoded,
            "diff": round(float(arr.mean()) - hardcoded, 3) if hardcoded else None,
        }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Recalculate RE24 from play-by-play data")
    parser.add_argument("--input", required=True,
                        help="Path to play_states CSV")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    output_dir = Path(args.input).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(output_dir / "re24_computed.json")

    print(f"Loading data from {args.input}")
    states = load_play_states(args.input)
    print(f"Loaded {len(states)} play states")

    print("Computing RE24 from actual data...\n")
    re24 = compute_re24(states)

    # Print comparison table
    print(f"{'State':<12} {'Computed':>9} {'Hardcoded':>10} "
          f"{'Diff':>8} {'StdDev':>8} {'N':>8}")
    print("=" * 62)

    total_abs_diff = 0
    n_compared = 0

    for label, vals in re24.items():
        computed = vals["computed_re"]
        hc = vals["hardcoded_re"]
        diff = vals["diff"]
        std = vals["std"]
        n = vals["count"]

        hc_str = f"{hc:.3f}" if hc is not None else "N/A"
        diff_str = f"{diff:+.3f}" if diff is not None else "N/A"

        print(f"  {label:<10} {computed:>9.3f} {hc_str:>10} "
              f"{diff_str:>8} {std:>8.3f} {n:>8}")

        if diff is not None:
            total_abs_diff += abs(diff)
            n_compared += 1

    if n_compared > 0:
        print(f"\nMean Absolute Difference (vs hardcoded): "
              f"{total_abs_diff / n_compared:.4f}")
        print(f"States compared: {n_compared}")

    # Save
    with open(output, "w") as f:
        json.dump(re24, f, indent=2)
    print(f"\nRE24 table saved to {output}")

    # Also output a Python-ready dict for updating win_probability.py
    py_output = output_dir / "re24_python_dict.txt"
    with open(py_output, "w") as f:
        f.write("# RE24 computed from play-by-play data\n")
        f.write("RE24_MLB = {\n")
        for label, vals in re24.items():
            s = tuple(vals["state"])
            f.write(f"    {s}: {vals['computed_re']:.3f},  "
                    f"# {label} (n={vals['count']})\n")
        f.write("}\n")
    print(f"Python dict saved to {py_output}")


if __name__ == "__main__":
    main()
