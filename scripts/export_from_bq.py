"""
Export play-by-play data from BigQuery to local CSVs.

Replaces slow MLB Stats API fetching with instant BQ export.
Falls back gracefully if BQ is unavailable.

Usage:
  python scripts/export_from_bq.py [--output-dir data/]
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


PROJECT = "data-platform-490901"
DATASET = "mlb_wp"
TABLE = "play_states"

FIELDNAMES = [
    "game_pk", "date", "home_team", "away_team", "home_won",
    "play_idx", "inning", "half_inning", "outs",
    "runner_1b", "runner_2b", "runner_3b",
    "home_score", "away_score", "score_diff",
    "home_score_after", "away_score_after", "event",
]


def export_from_bq(output_dir: Path) -> bool:
    """Export play_states from BigQuery, split by year into CSVs.

    Returns True if successful, False if BQ is unavailable.
    """
    try:
        from google.cloud import bigquery
    except ImportError:
        print("google-cloud-bigquery not installed")
        return False

    # Check for credentials
    sa_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_sa_key = os.environ.get("GCP_SA_KEY")

    if gcp_sa_key and not sa_key:
        # Write SA key to temp file (GitHub Actions pattern)
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(gcp_sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    try:
        client = bigquery.Client(project=PROJECT)

        # Get available years
        query = f"""
            SELECT DISTINCT CAST(SUBSTR(date, 1, 4) AS INT64) AS year
            FROM `{PROJECT}.{DATASET}.{TABLE}`
            ORDER BY year
        """
        years = [row.year for row in client.query(query).result()]
        print(f"BQ has data for years: {years}")

        total_rows = 0
        for year in years:
            csv_path = output_dir / f"play_states_{year}.csv"

            query = f"""
                SELECT *
                FROM `{PROJECT}.{DATASET}.{TABLE}`
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export play-by-play from BigQuery to CSV")
    parser.add_argument("--output-dir", default="data/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = export_from_bq(output_dir)
    if not success:
        print("BQ export failed — use fetch_game_states.py as fallback")
        sys.exit(1)


if __name__ == "__main__":
    main()
