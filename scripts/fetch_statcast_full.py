"""
Fetch full Statcast pitch-level data for WP model training.

Downloads ~7M rows (2015-2024) via pybaseball and saves to Google Drive
for BQ loading. Designed to run on Google Colab (colab-mcp).

Output: Google Drive kaggle/mlb_wp/statcast_{year}.parquet (one per year)

Usage (Colab):
  !pip install pybaseball pandas pyarrow
  %run fetch_statcast_full.py --year 2024 --output-dir /content/drive/MyDrive/kaggle/mlb_wp/

Usage (local, one month at a time):
  python scripts/fetch_statcast_full.py --year 2024 --month 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from pybaseball import statcast


# Columns to keep for WP model (drop unnecessary ones to save space)
WP_COLUMNS = [
    # Game state
    "game_pk", "game_date", "game_year", "game_type",
    "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score",
    "post_home_score", "post_away_score",
    # Players
    "batter", "pitcher", "player_name",
    "stand", "p_throws",
    # Pitch data
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    # Bat tracking (2024+ only — Hawk-Eye)
    "bat_speed", "swing_length", "attack_angle", "attack_direction",
    # Batted ball
    "type", "events", "description", "des",
    "bb_type",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "hit_distance_sc", "hc_x", "hc_y",
    # Expected stats
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
    "babip_value", "iso_value",
    "woba_value", "woba_denom",
    # Run/Win expectancy (MLB home_win_exp — benchmark)
    "home_win_exp",
    "bat_win_exp",
    "delta_home_win_exp",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    # Zone
    "zone", "sz_top", "sz_bot",
    # Context
    "if_fielding_alignment", "of_fielding_alignment",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
]


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
    df = statcast(start, end)
    elapsed = time.time() - t0
    print(f"  Raw: {len(df):,} pitches in {elapsed:.0f}s")

    if len(df) == 0:
        print(f"  WARNING: No data for {year}")
        return output_path

    # Filter to regular season + postseason
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
        print(f"  After game_type filter: {len(df):,}")

    # Keep only WP-relevant columns
    available = [c for c in WP_COLUMNS if c in df.columns]
    df = df[available].copy()

    # Save as parquet (much smaller than CSV)
    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {output_path} ({len(df):,} rows, {size_mb:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Statcast data for WP model training")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to fetch (default: all 2015-2024)")
    parser.add_argument("--month", type=int, default=None,
                        help="Single month (for incremental fetch)")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
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
