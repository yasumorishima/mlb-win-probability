"""
Fetch full Statcast pitch-level data for WP model training.

Downloads ALL columns (~118) from pybaseball for 2015-2025.
Saves as yearly parquet files for BQ loading.

Output: data/statcast/statcast_{year}.parquet (one per year)

Usage (GitHub Actions):
  python scripts/fetch_statcast_full.py --start-year 2015 --end-year 2025

Usage (single year):
  python scripts/fetch_statcast_full.py --year 2024 --output-dir data/statcast/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from pybaseball import statcast


def fetch_year(year: int, output_dir: Path, month: int | None = None) -> Path:
    """Fetch one year (or month) of Statcast data."""
    if month:
        start = f"{year}-{month:02d}-01"
        if month == 12:
            end = f"{year}-12-31"
        else:
            end = f"{year}-{month + 1:02d}-01"
        fname = f"statcast_{year}_{month:02d}.parquet"
    else:
        start = f"{year}-03-15"
        end = f"{year}-11-15"
        fname = f"statcast_{year}.parquet"

    output_path = output_dir / fname

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        print(f"  {fname} already exists ({len(existing):,} rows), skipping")
        return output_path

    print(f"  Fetching {start} to {end}...")
    t0 = time.time()

    # pybaseball fetches in weekly chunks internally
    # Retry up to 3 times for transient API/parse errors
    max_retries = 3
    df = None
    for attempt in range(1, max_retries + 1):
        try:
            df = statcast(start, end)
            break
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 30 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to fetch {year} after {max_retries} attempts: {e}"
                ) from e

    elapsed = time.time() - t0
    print(f"  Raw: {len(df):,} pitches in {elapsed:.0f}s")

    if len(df) == 0:
        print(f"  WARNING: No data for {year}")
        return output_path

    # Filter to regular season + postseason
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
        print(f"  After game_type filter: {len(df):,}")

    # Keep ALL columns — no filtering
    print(f"  Columns: {len(df.columns)}")

    # Save as parquet (much smaller than CSV)
    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path} ({len(df):,} rows, {len(df.columns)} cols, {size_mb:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Statcast data for WP model training")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to fetch (default: all 2015-2024)")
    parser.add_argument("--month", type=int, default=None,
                        help="Single month (for incremental fetch)")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", default="data/statcast/",
                        help="Output directory for parquet files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))

    print(f"Fetching Statcast data: {years}")
    print(f"Output: {output_dir}")
    print()

    total_rows = 0
    for year in years:
        print(f"\n{'=' * 50}")
        print(f"Year: {year}")
        print(f"{'=' * 50}")
        path = fetch_year(year, output_dir, args.month)
        if path.exists():
            df = pd.read_parquet(path)
            total_rows += len(df)

        # Rate limiting between years
        if len(years) > 1:
            time.sleep(5)

    print(f"\n{'=' * 50}")
    print(f"TOTAL: {total_rows:,} pitches across {len(years)} years")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
