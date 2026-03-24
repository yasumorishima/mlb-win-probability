"""Load MLB play-by-play CSV data into BigQuery."""

import os
import sys
from pathlib import Path

from google.cloud import bigquery

PROJECT = "data-platform-490901"
DATASET = "mlb_wp"
DATA_DIR = Path(__file__).parent.parent / "data"

# Ensure credentials are set
KEY_PATH = r"C:\Users\fw_ya\Downloads\data-platform-490901-46cfc6902165.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH


def get_schema():
    """Schema matching fetch_game_states.py FIELDNAMES."""
    return [
        bigquery.SchemaField("game_pk", "INTEGER"),
        bigquery.SchemaField("date", "STRING"),
        bigquery.SchemaField("home_team", "STRING"),
        bigquery.SchemaField("away_team", "STRING"),
        bigquery.SchemaField("home_won", "INTEGER"),
        bigquery.SchemaField("play_idx", "INTEGER"),
        bigquery.SchemaField("inning", "INTEGER"),
        bigquery.SchemaField("half_inning", "STRING"),
        bigquery.SchemaField("outs", "INTEGER"),
        bigquery.SchemaField("runner_1b", "INTEGER"),
        bigquery.SchemaField("runner_2b", "INTEGER"),
        bigquery.SchemaField("runner_3b", "INTEGER"),
        bigquery.SchemaField("home_score", "INTEGER"),
        bigquery.SchemaField("away_score", "INTEGER"),
        bigquery.SchemaField("score_diff", "INTEGER"),
        bigquery.SchemaField("home_score_after", "INTEGER"),
        bigquery.SchemaField("away_score_after", "INTEGER"),
        bigquery.SchemaField("event", "STRING"),
    ]


def main():
    client = bigquery.Client(project=PROJECT)

    # Create dataset if not exists
    dataset_ref = f"{PROJECT}.{DATASET}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset {DATASET} ready")

    # Find all play_states CSV files
    csv_files = sorted(DATA_DIR.glob("play_states_*.csv"))
    if not csv_files:
        print(f"ERROR: No play_states_*.csv files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    schema = get_schema()
    total_rows = 0

    for csv_path in csv_files:
        year = csv_path.stem.replace("play_states_", "")
        table_id = f"{PROJECT}.{DATASET}.play_states_{year}"

        print(f"\nLoading {csv_path.name} -> {table_id}")

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            skip_leading_rows=1,  # CSV header
            source_format=bigquery.SourceFormat.CSV,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        with open(csv_path, "rb") as f:
            job = client.load_table_from_file(f, table_id, job_config=job_config)

        job.result()  # Wait for completion

        table = client.get_table(table_id)
        print(f"  -> {table.num_rows:,} rows loaded")
        total_rows += table.num_rows

    # Also create a unified view across all years
    years = [f.stem.replace("play_states_", "") for f in csv_files]
    union_parts = [
        f"SELECT * FROM `{PROJECT}.{DATASET}.play_states_{y}`" for y in years
    ]
    view_sql = " UNION ALL ".join(union_parts)
    view_id = f"{PROJECT}.{DATASET}.play_states_all"

    view = bigquery.Table(view_id)
    view.view_query = view_sql
    try:
        client.delete_table(view_id, not_found_ok=True)
        client.create_table(view)
        print(f"\nView {view_id} created (union of {len(years)} tables)")
    except Exception as e:
        print(f"\nWARN: Could not create view: {e}")

    # Verify with COUNT queries
    print("\n=== Verification ===")
    for csv_path in csv_files:
        year = csv_path.stem.replace("play_states_", "")
        table_id = f"{PROJECT}.{DATASET}.play_states_{year}"
        query = f"SELECT COUNT(*) as cnt FROM `{table_id}`"
        result = list(client.query(query).result())
        count = result[0].cnt
        print(f"  play_states_{year}: {count:,} rows")

    # Total from view
    try:
        query = f"SELECT COUNT(*) as cnt FROM `{view_id}`"
        result = list(client.query(query).result())
        count = result[0].cnt
        print(f"  play_states_all (view): {count:,} rows")
    except Exception:
        pass

    print(f"\nTotal rows loaded: {total_rows:,}")
    print("Done!")


if __name__ == "__main__":
    main()
