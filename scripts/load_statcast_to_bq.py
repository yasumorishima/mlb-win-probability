"""
Load Statcast parquet files into BigQuery for WP model training.

Loads year-by-year to avoid OOM (each year ~800MB, runner has 7GB RAM).
First year uses WRITE_TRUNCATE, subsequent years WRITE_APPEND.

Reads parquet files from data/statcast/ and loads into
data-platform-490901.mlb_wp.statcast_pitches table.

Usage:
  python scripts/load_statcast_to_bq.py --data-dir data/statcast/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT = "data-platform-490901"
DATASET = "mlb_wp"
TABLE = "statcast_pitches"


def _add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns for WP model."""
    if "home_score" in df.columns and "away_score" in df.columns:
        df["score_diff"] = df["home_score"] - df["away_score"]
    if "inning_topbot" in df.columns:
        df["is_bottom"] = (df["inning_topbot"] == "Bot").astype(int)
    if all(c in df.columns for c in ["on_1b", "on_2b", "on_3b"]):
        df["total_runners"] = (
            df["on_1b"].notna().astype(int)
            + df["on_2b"].notna().astype(int)
            + df["on_3b"].notna().astype(int)
        )
        df["scoring_position"] = (
            df["on_2b"].notna().astype(int)
            + df["on_3b"].notna().astype(int)
        )
    return df


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert types for BQ compatibility."""
    # String columns — force to str to avoid Int64/Arrow type errors
    str_cols = [
        "sv_id", "game_date", "des", "description",
        "pitch_type", "pitch_name", "events", "bb_type", "type",
        "home_team", "away_team", "player_name", "stand", "p_throws",
        "inning_topbot", "game_type", "umpire",
        "if_fielding_alignment", "of_fielding_alignment",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", None)

    # Convert numeric columns
    numeric_cols = [
        "inning", "outs_when_up", "balls", "strikes",
        "home_score", "away_score", "bat_score", "fld_score",
        "post_home_score", "post_away_score",
        "launch_angle", "hit_distance_sc",
        "release_spin_rate", "spin_axis",
        "bat_speed", "swing_length", "attack_angle",
        "zone", "score_diff", "is_bottom",
        "total_runners", "scoring_position",
        "post_bat_score", "post_fld_score",
        "bat_score_diff", "hit_location",
        "age_bat", "age_pit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Downcast nullable Int64 columns to float64 (Arrow compatibility)
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].astype("float64")

    return df


def load_to_bq(data_dir: Path, append: bool = False):
    """Load all statcast parquet files into BigQuery, one year at a time.

    Args:
        data_dir: Directory with statcast_*.parquet files.
        append: If True, use WRITE_APPEND for all files (don't truncate).
                Use this when adding years to an existing table.
    """
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
    table_ref = f"{PROJECT}.{DATASET}.{TABLE}"

    # Coverage tracking
    coverage = []
    null_rates = {}  # {year: {col: null_pct}}
    total_rows = 0
    total_cols = 0
    all_columns = set()

    # Key columns we specifically want to track
    key_cols = [
        # Fielder IDs (should exist all years)
        "fielder_2", "fielder_3", "fielder_4", "fielder_5",
        "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        # Batted ball detail
        "hit_location", "des", "bb_type",
        "launch_speed", "launch_angle", "hit_distance_sc",
        # Expected stats
        "estimated_woba_using_speedangle", "woba_value", "babip_value", "iso_value",
        # Pitcher stats
        "delta_pitcher_run_exp", "delta_run_exp",
        # Bat tracking (2024+ only — expected NULL for earlier years)
        "bat_speed", "swing_length", "hyper_speed",
        # Player info
        "age_bat", "age_pit", "umpire",
        # WP benchmark
        "home_win_exp",
    ]

    # Load year-by-year to avoid OOM
    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf)
        year_label = pf.stem.replace("statcast_", "")
        n_raw = len(df)
        n_cols = len(df.columns)
        total_cols = max(total_cols, n_cols)
        all_columns.update(df.columns.tolist())

        print(f"\n{'='*50}")
        print(f"  {pf.name}: {n_raw:,} rows, {n_cols} columns")

        # Null rate per key column
        yr_nulls = {}
        for col in key_cols:
            if col in df.columns:
                null_pct = df[col].isna().mean() * 100
                yr_nulls[col] = null_pct
            else:
                yr_nulls[col] = -1  # column missing entirely
        null_rates[year_label] = yr_nulls

        # Flag columns that are 100% null (data quality issue)
        full_null = [c for c, v in yr_nulls.items() if v == 100.0]
        if full_null:
            print(f"  WARNING: 100% null columns: {full_null}")

        # Add computed columns
        df = _add_computed_columns(df)
        df = _convert_types(df)

        n_final = len(df)
        total_rows += n_final

        # First file: TRUNCATE, rest: APPEND
        # ALLOW_FIELD_ADDITION: 2024+ has bat_speed/swing_length etc. not in 2015
        if append:
            disposition = "WRITE_APPEND"
        else:
            disposition = "WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND"

        job_config = bigquery.LoadJobConfig(
            write_disposition=disposition,
            autodetect=True,
        )
        # ALLOW_FIELD_ADDITION only valid with WRITE_APPEND
        if disposition == "WRITE_APPEND":
            job_config.schema_update_options = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
            ]

        print(f"  Loading to BQ ({disposition})...")
        job = client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        job.result()  # Wait for completion
        print(f"  Done: {n_final:,} rows loaded")

        present = [c for c in key_cols if c in df.columns]
        missing = [c for c in key_cols if c not in df.columns]

        coverage.append({
            "year": year_label,
            "rows": n_final,
            "columns": len(df.columns),
            "key_present": len(present),
            "key_missing": missing,
        })

        # Free memory
        del df

    # ============================================================
    # COVERAGE REPORT
    # ============================================================
    n_key = len(key_cols)
    print(f"\n{'='*60}")
    print("COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"{'Year':<8} {'Rows':>10} {'Cols':>6} {'Key':>6}")
    print(f"{'-'*8} {'-'*10} {'-'*6} {'-'*6}")
    for c in coverage:
        print(f"{c['year']:<8} {c['rows']:>10,} {c['columns']:>6} "
              f"{c['key_present']:>3}/{n_key}")
    print(f"{'-'*8} {'-'*10} {'-'*6} {'-'*6}")
    print(f"{'TOTAL':<8} {total_rows:>10,} {total_cols:>6}")

    # Show which key columns are missing (across any year)
    all_missing = set()
    for c in coverage:
        all_missing.update(c["key_missing"])
    if all_missing:
        print(f"\nKey columns missing in some years: {sorted(all_missing)}")

    # ============================================================
    # NULL RATE REPORT (per key column per year)
    # ============================================================
    years_sorted = sorted(null_rates.keys())
    print(f"\n{'='*60}")
    print("NULL RATE REPORT (key columns, % null)")
    print(f"{'='*60}")

    # Header
    header = f"{'Column':<35}" + "".join(f"{y:>7}" for y in years_sorted)
    print(header)
    print("-" * len(header))

    for col in key_cols:
        row = f"{col:<35}"
        any_issue = False
        for yr in years_sorted:
            val = null_rates[yr].get(col, -1)
            if val == -1:
                row += "   MISS"
                any_issue = True
            elif val == 100.0:
                row += "  100.0"
                any_issue = True
            elif val > 50:
                row += f"  {val:5.1f}"
                any_issue = True
            else:
                row += f"  {val:5.1f}"
        if any_issue:
            row += "  ⚠"
        print(row)

    # Summary: total columns across all years
    print(f"\nTotal unique columns across all years: {len(all_columns)}")

    # ============================================================
    # Verify BQ table
    # ============================================================
    table = client.get_table(table_ref)
    print(f"\nBQ Table: {table.num_rows:,} rows, {len(table.schema)} columns, "
          f"{table.num_bytes / 1024**3:.2f} GB")

    return coverage


def main():
    parser = argparse.ArgumentParser(
        description="Load Statcast parquets into BigQuery")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with statcast_*.parquet files")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing BQ table (don't truncate)")
    args = parser.parse_args()

    load_to_bq(Path(args.data_dir), append=args.append)


if __name__ == "__main__":
    main()
