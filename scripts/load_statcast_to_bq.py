"""
Load Statcast parquet files into BigQuery for WP model training.

Reads parquet files from Google Drive (or local) and loads into
data-platform-490901.mlb_wp.statcast_pitches table.

Usage:
  python scripts/load_statcast_to_bq.py --data-dir data/statcast/
  python scripts/load_statcast_to_bq.py --data-dir /content/drive/MyDrive/kaggle/mlb_wp/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT = "data-platform-490901"
DATASET = "mlb_wp"
TABLE = "statcast_pitches"


def load_to_bq(data_dir: Path):
    """Load all statcast parquet files into BigQuery."""
    from google.cloud import bigquery

    # Handle GCP auth
    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    client = bigquery.Client(project=PROJECT)

    # Find parquet files
    parquet_files = sorted(data_dir.glob("statcast_*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Load and concatenate
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        print(f"  {pf.name}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined):,} rows, {len(combined.columns)} columns")

    # Add computed columns for WP model
    combined["score_diff"] = combined["home_score"] - combined["away_score"]
    combined["is_bottom"] = (combined["inning_topbot"] == "Bot").astype(int)
    combined["total_runners"] = (
        combined["on_1b"].notna().astype(int) +
        combined["on_2b"].notna().astype(int) +
        combined["on_3b"].notna().astype(int)
    )
    combined["scoring_position"] = (
        combined["on_2b"].notna().astype(int) +
        combined["on_3b"].notna().astype(int)
    )

    # Convert nullable int columns
    int_cols = ["inning", "outs_when_up", "balls", "strikes",
                "home_score", "away_score", "bat_score", "fld_score",
                "post_home_score", "post_away_score",
                "launch_angle", "hit_distance_sc",
                "release_spin_rate", "spin_axis",
                "bat_speed", "swing_length", "attack_angle",
                "zone", "score_diff", "is_bottom",
                "total_runners", "scoring_position"]
    for col in int_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Convert game_date to string for BQ
    if "game_date" in combined.columns:
        combined["game_date"] = combined["game_date"].astype(str)

    # Load to BigQuery
    table_ref = f"{PROJECT}.{DATASET}.{TABLE}"
    print(f"\nLoading to {table_ref}...")

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(
        combined, table_ref, job_config=job_config
    )
    job.result()  # Wait for completion

    table = client.get_table(table_ref)
    print(f"Loaded: {table.num_rows:,} rows, {table.num_bytes / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Load Statcast parquets into BigQuery")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with statcast_*.parquet files")
    args = parser.parse_args()

    load_to_bq(Path(args.data_dir))


if __name__ == "__main__":
    main()
