"""
Colab-MCP用: Statcastデータ取得 → Drive保存 → BQロード → Drive削除

Google Colabで実行することを想定。
1. pybaseballでStatcastデータを年ごとに取得（2015-2025）
2. Google Drive に一時保存（parquet）— 全カラム保持（フィルタなし）
3. BigQueryにロード
4. Drive上のparquetを削除

Usage (colab-mcp):
  !pip install pybaseball pyarrow google-cloud-bigquery savant-extras
  %run colab_fetch_statcast.py
"""

import time
from pathlib import Path

import pandas as pd
from pybaseball import statcast

# ============================================================
# Config
# ============================================================

YEARS = list(range(2015, 2026))  # 2015-2025
DRIVE_DIR = Path("/content/drive/MyDrive/kaggle/mlb_wp/statcast/")
BQ_PROJECT = "data-platform-490901"
BQ_DATASET = "mlb_wp"
BQ_TABLE = "statcast_pitches"

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

    # Keep ALL columns — no filtering
    print(f"  [{year}] Columns: {len(df.columns)}")

    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [{year}] Saved: {len(df):,} rows, {size_mb:.1f} MB")

    return output_path


def load_to_bq():
    """Load all parquet files from Drive into BigQuery, year-by-year (OOM対策)."""
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

    print(f"\nLoading {len(parquet_files)} files to BigQuery (year-by-year)...")

    total_rows = 0
    max_cols = 0

    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf)
        n_raw = len(df)
        n_cols = len(df.columns)
        max_cols = max(max_cols, n_cols)

        # Add computed columns
        if "home_score" in df.columns and "away_score" in df.columns:
            df["score_diff"] = df["home_score"] - df["away_score"]
        if "inning_topbot" in df.columns:
            df["is_bottom"] = (df["inning_topbot"] == "Bot").astype(int)

        # Convert game_date to string
        if "game_date" in df.columns:
            df["game_date"] = df["game_date"].astype(str)

        # Convert numeric columns
        numeric_cols = [
            "inning", "outs_when_up", "balls", "strikes",
            "home_score", "away_score", "bat_score", "fld_score",
            "post_home_score", "post_away_score",
            "launch_angle", "hit_distance_sc",
            "release_spin_rate", "spin_axis",
            "bat_speed", "swing_length", "attack_angle",
            "zone", "score_diff", "is_bottom",
            "hit_location", "age_bat", "age_pit",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # First file: TRUNCATE, rest: APPEND
        # ALLOW_FIELD_ADDITION: 2024+ has bat_speed/swing_length etc. not in 2015
        disposition = "WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND"
        job_config = bigquery.LoadJobConfig(
            write_disposition=disposition,
            autodetect=True,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
            ],
        )

        print(f"  {pf.name}: {n_raw:,} rows, {n_cols} cols ({disposition})")
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        total_rows += len(df)

        del df  # Free memory

    table = client.get_table(table_ref)
    print(f"\n  BQ loaded: {table.num_rows:,} rows, {len(table.schema)} cols, "
          f"{table.num_bytes / 1024**3:.2f} GB")
    print(f"  Max columns per year: {max_cols}")
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
