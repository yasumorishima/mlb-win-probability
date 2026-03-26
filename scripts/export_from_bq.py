"""
Export play-by-play data from BigQuery to local CSVs.

Exports two datasets:
  1. play_states — all game state transitions (for v1/v2/ensemble evaluation)
  2. statcast_pitches — at-bat outcomes with pitch/hit features (for Statcast model)

Usage:
  python scripts/export_from_bq.py [--output-dir data/]
  python scripts/export_from_bq.py --output-dir data/ --statcast
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path


def _log_elapsed(label: str, start: float, budget_min: int = 120):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")


PROJECT = "data-platform-490901"
DATASET_WP = "mlb_wp"             # play_states (WP-specific)
DATASET_SHARED = "mlb_shared"     # statcast_pitches (shared via mlb-data-pipeline)
TABLE = "play_states"
STATCAST_TABLE = "statcast_pitches"

FIELDNAMES = [
    "game_pk", "date", "home_team", "away_team", "home_won",
    "play_idx", "inning", "half_inning", "outs",
    "runner_1b", "runner_2b", "runner_3b",
    "home_score", "away_score", "score_diff",
    "home_score_after", "away_score_after", "event",
]


def _get_bq_client():
    """Get authenticated BigQuery client."""
    from google.cloud import bigquery

    sa_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_sa_key = os.environ.get("GCP_SA_KEY")

    if gcp_sa_key and not sa_key:
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(gcp_sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    return bigquery.Client(project=PROJECT)


def export_from_bq(output_dir: Path) -> bool:
    """Export play_states from BigQuery, split by year into CSVs.

    Returns True if successful, False if BQ is unavailable.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("google-cloud-bigquery not installed")
        return False

    try:
        client = _get_bq_client()

        # Get available years
        query = f"""
            SELECT DISTINCT CAST(SUBSTR(date, 1, 4) AS INT64) AS year
            FROM `{PROJECT}.{DATASET_WP}.{TABLE}`
            ORDER BY year
        """
        years = [row.year for row in client.query(query).result()]
        print(f"BQ has data for years: {years}")

        total_rows = 0
        for year in years:
            csv_path = output_dir / f"play_states_{year}.csv"

            query = f"""
                SELECT *
                FROM `{PROJECT}.{DATASET_WP}.{TABLE}`
                WHERE STARTS_WITH(date, '{year}')
                ORDER BY game_pk, play_idx
            """
            rows = list(client.query(query).result())

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: getattr(row, k, "") for k in FIELDNAMES})

            print(f"  {year}: {len(rows):,} plays -> {csv_path}")
            total_rows += len(rows)

        print(f"Total: {total_rows:,} plays exported from BigQuery")
        return True

    except Exception as e:
        print(f"BQ export failed: {e}")
        return False


def export_statcast_from_bq(output_dir: Path) -> bool:
    """Export statcast_pitches (at-bat outcomes) from BigQuery, split by year.

    Exports all columns needed by engineer_features() in train_wp_statcast.py.
    Used by ensemble and Bayesian training to provide real Statcast features.

    Returns True if successful, False if BQ is unavailable.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("google-cloud-bigquery not installed")
        return False

    try:
        client = _get_bq_client()

        query = f"""
            SELECT DISTINCT game_year
            FROM `{PROJECT}.{DATASET_SHARED}.{STATCAST_TABLE}`
            WHERE game_type = 'R' AND events IS NOT NULL
            ORDER BY game_year
        """
        years = [int(row.game_year) for row in client.query(query).result()]
        print(f"BQ statcast_pitches years: {years}")

        total_rows = 0
        for year in years:
            csv_path = output_dir / f"statcast_pitches_{year}.csv"

            query = f"""
                SELECT *
                FROM `{PROJECT}.{DATASET_SHARED}.{STATCAST_TABLE}`
                WHERE game_type = 'R' AND events IS NOT NULL
                  AND game_year = {year}
                ORDER BY game_pk, inning, is_bottom, outs_when_up
            """
            df = client.query(query).to_dataframe()
            df.to_csv(csv_path, index=False)
            print(f"  {year}: {len(df):,} at-bats -> {csv_path}")
            total_rows += len(df)

        print(f"Total: {total_rows:,} at-bat outcomes exported from BigQuery")
        return True

    except Exception as e:
        print(f"BQ statcast export failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export play-by-play from BigQuery to CSV")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--statcast", action="store_true",
                        help="Also export statcast_pitches (at-bat outcomes)")
    args = parser.parse_args()

    t0 = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = export_from_bq(output_dir)
    if not success:
        print("BQ export failed — use fetch_game_states.py as fallback")
        sys.exit(1)
    _log_elapsed("export_play_states", t0)

    if args.statcast:
        statcast_ok = export_statcast_from_bq(output_dir)
        if not statcast_ok:
            print("WARNING: Statcast export failed (ensemble will use game-state fallback)")
        _log_elapsed("export_statcast", t0)


if __name__ == "__main__":
    main()
