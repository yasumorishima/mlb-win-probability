"""
Fetch Stuff+/Location+/Pitching+ from FanGraphs and load to BigQuery.

Creates mlb_wp.pitcher_plus_stats table with:
  - pitcher_id, season, Stuff+, Location+, Pitching+
  - Per-pitch-type: Stf+/Loc+/Pit+ for FA/SI/SL/CH/CU/FC/FS/KC

Usage:
  python scripts/fetch_pitcher_plus.py --start-year 2020 --end-year 2024
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from pybaseball import pitching_stats


# Columns to extract
OVERALL_COLS = ["Stuff+", "Location+", "Pitching+"]
PITCH_TYPES = ["FA", "SI", "SL", "CH", "CU", "FC", "FS", "KC"]
PER_PITCH_COLS = []
for pt in PITCH_TYPES:
    PER_PITCH_COLS.extend([f"Stf+ {pt}", f"Loc+ {pt}", f"Pit+ {pt}"])

BQ_PROJECT = "data-platform-490901"
BQ_DATASET = "mlb_wp"
BQ_TABLE = "pitcher_plus_stats"


def fetch_all_years(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch FanGraphs pitching stats with Stuff+/Location+/Pitching+."""
    frames = []

    for year in range(start, end + 1):
        print(f"  [{year}] Fetching FanGraphs pitching stats...")
        t0 = time.time()

        max_retries = 3
        df = None
        for attempt in range(1, max_retries + 1):
            try:
                df = pitching_stats(year, year, qual=30)
                break
            except Exception as e:
                print(f"    Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(10 * attempt)
                else:
                    print(f"    Skipping {year}")

        if df is None or len(df) == 0:
            continue

        elapsed = time.time() - t0

        # Extract relevant columns
        keep = ["IDfg", "Name"]
        keep += [c for c in OVERALL_COLS if c in df.columns]
        keep += [c for c in PER_PITCH_COLS if c in df.columns]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["season"] = year

        # Keep IDfg for MLBAM mapping (renamed later)

        frames.append(df)
        print(f"    {len(df)} pitchers, {len(df.columns)} cols in {elapsed:.1f}s")
        time.sleep(2)

    if not frames:
        print("ERROR: No data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Map FanGraphs ID → MLBAM ID (Statcast uses MLBAM)
    print("\n  Mapping FanGraphs ID → MLBAM ID via Chadwick register...")
    from pybaseball import chadwick_register
    reg = chadwick_register()
    fg_to_mlbam = dict(zip(
        reg["key_fangraphs"].astype(int),
        reg["key_mlbam"].astype(int),
    ))
    # IDfg may be int or str depending on pybaseball version
    combined["pitcher_id"] = combined["IDfg"].astype(int).map(fg_to_mlbam)
    n_mapped = combined["pitcher_id"].notna().sum()
    n_total = len(combined)
    print(f"  Mapped: {n_mapped}/{n_total} ({n_mapped/n_total*100:.1f}%)")
    combined = combined.dropna(subset=["pitcher_id"])
    combined["pitcher_id"] = combined["pitcher_id"].astype(int)

    # Clean column names for BQ (no spaces, no +)
    col_map = {}
    for c in combined.columns:
        new = c.lower().replace("+", "_plus").replace(" ", "_").replace("/", "_")
        col_map[c] = new
    combined = combined.rename(columns=col_map)

    # Drop FanGraphs ID (keep only MLBAM pitcher_id)
    combined = combined.drop(columns=["idfg"], errors="ignore")

    output_path = output_dir / "pitcher_plus_stats.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")

    return combined


def load_to_bq(df: pd.DataFrame):
    """Load pitcher plus stats to BigQuery."""
    from google.cloud import bigquery

    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    client = bigquery.Client(project=BQ_PROJECT)
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    print(f"BQ: {table.num_rows:,} rows, {len(table.schema)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Stuff+/Location+/Pitching+ from FanGraphs → BQ")
    parser.add_argument("--start-year", type=int, default=2020,
                        help="Start year (Stuff+ available from 2020)")
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for CSV")
    parser.add_argument("--no-bq", action="store_true",
                        help="Skip BQ upload (CSV only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Stuff+/Location+/Pitching+ ({args.start_year}-{args.end_year})")
    df = fetch_all_years(args.start_year, args.end_year, output_dir)

    if len(df) == 0:
        return

    # Coverage report
    print(f"\n{'='*50}")
    print("COVERAGE REPORT")
    print(f"{'='*50}")
    print(f"Years: {df['season'].nunique()}")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print()
    for col in df.columns:
        if col not in ("fangraphs_id", "name", "season"):
            null_pct = df[col].isna().mean() * 100
            print(f"  {col}: null={null_pct:.1f}%, mean={df[col].mean():.1f}")

    if not args.no_bq:
        print(f"\nLoading to BQ...")
        load_to_bq(df)


if __name__ == "__main__":
    main()
