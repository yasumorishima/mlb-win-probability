"""
Colab-MCP用: Statcastデータ取得 → Drive保存 → BQロード → Drive削除

Google Colabで実行することを想定。
1. pybaseballでStatcastデータを年ごとに取得（2015-2024）
2. Google Drive に一時保存（parquet）
3. BigQueryにロード
4. Drive上のparquetを削除

Usage (colab-mcp):
  !pip install pybaseball pyarrow google-cloud-bigquery savant-extras
  %run colab_fetch_statcast.py
"""

import os
import time
from pathlib import Path

import pandas as pd
from pybaseball import statcast

# ============================================================
# Config
# ============================================================

YEARS = list(range(2015, 2025))  # 2015-2024
DRIVE_DIR = Path("/content/drive/MyDrive/kaggle/mlb_wp/statcast/")
BQ_PROJECT = "data-platform-490901"
BQ_DATASET = "mlb_wp"
BQ_TABLE = "statcast_pitches"

# Columns to keep (70 WP-relevant columns out of 118)
WP_COLUMNS = [
    "game_pk", "game_date", "game_year", "game_type",
    "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score",
    "post_home_score", "post_away_score",
    "batter", "pitcher", "player_name",
    "stand", "p_throws",
    "pitch_type", "pitch_name",
    "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    # Bat tracking (2024+ only)
    "bat_speed", "swing_length", "attack_angle", "attack_direction",
    # Batted ball
    "type", "events", "description",
    "bb_type",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "hit_distance_sc", "hc_x", "hc_y",
    # Expected stats
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value", "woba_denom",
    # Win expectancy (MLB official — benchmark)
    "home_win_exp", "bat_win_exp",
    "delta_home_win_exp", "delta_run_exp",
    # Zone
    "zone", "sz_top", "sz_bot",
    # Context
    "if_fielding_alignment", "of_fielding_alignment",
    "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
]


def fetch_and_save(year: int) -> Path | None:
    """Fetch one year of Statcast data and save to Drive."""
    output_path = DRIVE_DIR / f"statcast_{year}.parquet"

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        print(f"  [{year}] Already exists ({len(existing):,} rows), skipping")
        return output_path

    print(f"  [{year}] Fetching {year}-03-15 to {year}-11-15...")
    t0 = time.time()
    df = statcast(f"{year}-03-15", f"{year}-11-15")
    elapsed = time.time() - t0
    print(f"  [{year}] Raw: {len(df):,} pitches in {elapsed:.0f}s")

    if len(df) == 0:
        print(f"  [{year}] WARNING: No data")
        return None

    # Filter regular season + postseason
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])].copy()

    # Keep WP-relevant columns only
    available = [c for c in WP_COLUMNS if c in df.columns]
    df = df[available].copy()

    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [{year}] Saved: {len(df):,} rows, {size_mb:.1f} MB")

    return output_path


def load_to_bq():
    """Load all parquet files from Drive into BigQuery."""
    from google.cloud import bigquery

    # Auth (Colab uses default credentials from google.colab.auth)
    try:
        from google.colab import auth
        auth.authenticate_user()
    except ImportError:
        pass  # Not on Colab, use env var

    client = bigquery.Client(project=BQ_PROJECT)
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    parquet_files = sorted(DRIVE_DIR.glob("statcast_*.parquet"))
    if not parquet_files:
        print("ERROR: No parquet files found")
        return False

    print(f"\nLoading {len(parquet_files)} files to BigQuery...")

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        print(f"  {pf.name}: {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Add computed columns
    combined["score_diff"] = combined["home_score"] - combined["away_score"]
    combined["is_bottom"] = (combined["inning_topbot"] == "Bot").astype(int)

    # Convert game_date to string
    if "game_date" in combined.columns:
        combined["game_date"] = combined["game_date"].astype(str)

    print(f"  Total: {len(combined):,} rows, {len(combined.columns)} columns")

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(combined, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    print(f"  BQ loaded: {table.num_rows:,} rows, {table.num_bytes / 1024 / 1024:.1f} MB")
    return True


def cleanup_drive():
    """Delete parquet files from Drive after BQ load."""
    parquet_files = list(DRIVE_DIR.glob("statcast_*.parquet"))
    for pf in parquet_files:
        pf.unlink()
        print(f"  Deleted: {pf.name}")
    print(f"  Cleaned up {len(parquet_files)} files from Drive")


def main():
    # Mount Drive (Colab)
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass  # Not on Colab

    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch all years
    print("=" * 60)
    print("STEP 1: Fetch Statcast data (pybaseball)")
    print("=" * 60)
    for year in YEARS:
        fetch_and_save(year)
        time.sleep(3)  # Rate limiting

    # Step 2: Load to BigQuery
    print("\n" + "=" * 60)
    print("STEP 2: Load to BigQuery")
    print("=" * 60)
    success = load_to_bq()

    # Step 3: Cleanup Drive
    if success:
        print("\n" + "=" * 60)
        print("STEP 3: Cleanup Drive")
        print("=" * 60)
        cleanup_drive()
    else:
        print("\nBQ load failed, keeping Drive files for retry")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
