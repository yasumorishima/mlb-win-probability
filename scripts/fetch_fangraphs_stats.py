"""
Fetch FanGraphs season-level batting and pitching stats for WP model enrichment.

Downloads batting_stats and pitching_stats from FanGraphs via pybaseball,
maps FanGraphs IDs to MLBAM IDs (Statcast key), and saves CSV files.

These season-level stats (wRC+, SIERA, Stuff+, K%, etc.) are joined to
pitch-level Statcast data in train_wp_statcast.py to provide batter/pitcher
quality context that pitch-level data alone cannot capture.

Usage:
  python scripts/fetch_fangraphs_stats.py --start-year 2015 --end-year 2024
  python scripts/fetch_fangraphs_stats.py --no-bq   # CSV only
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

# =====================================================================
# Feature definitions — curated for WP model impact
# =====================================================================

# Batter features: season-level quality indicators
# These complement pitch-level Statcast (launch_speed, xwOBA per AB)
# with season-level context (plate discipline, overall quality).
BATTER_FEATURES = [
    # Core (all 0% null, 2015-2024)
    "wOBA", "xwOBA", "wRC+", "OPS", "AVG", "OBP", "SLG", "ISO", "BABIP",
    # Plate discipline (all 0% null)
    "K%", "BB%", "O-Swing%", "Z-Swing%", "O-Contact%", "Z-Contact%",
    "Zone%", "SwStr%", "Contact%",
    # Batted ball (all 0% null)
    "GB%", "FB%", "LD%", "IFFB%", "HR/FB",
    "Pull%", "Cent%", "Oppo%",
    "Soft%", "Med%", "Hard%", "HardHit%",
    # Value (all 0% null)
    "WAR", "Off", "Def", "BsR", "Spd",
    "Clutch", "WPA", "wRAA",
    # Pitch type values
    "wFB/C", "wSL/C", "wCH/C",
    # Context
    "G", "PA",
]

# Pitcher features: season-level quality indicators
PITCHER_FEATURES = [
    # Core ERA models (all 0% null)
    "ERA", "FIP", "xFIP", "SIERA",
    "ERA-", "FIP-", "xFIP-",
    # Rate (all 0% null)
    "K%", "BB%", "K-BB%", "K/9", "BB/9", "K/BB",
    "HR/9", "HR/FB", "WHIP", "BABIP", "LOB%",
    # Pitch quality (all 0% null)
    "SwStr%", "CSW%",
    "O-Swing%", "Z-Swing%", "O-Contact%", "Z-Contact%", "Zone%",
    "F-Strike%",
    # Stuff+/Location+/Pitching+ (2020+, 100% null for 2015-2019)
    "Stuff+", "Location+", "Pitching+",
    # Batted ball (all 0% null)
    "GB%", "FB%", "LD%", "IFFB%",
    "Pull%", "Cent%", "Oppo%",
    "Soft%", "Med%", "Hard%",
    # Pitch type values (wSL/C 7.4% null, wCH/C 32.9% null — pitchers who don't throw that type)
    "wFB/C", "wSL/C", "wCH/C",
    # Role (Start-IP 40.9% null, Relief-IP 30.8% null — role-specific)
    "GS", "Start-IP", "Relief-IP",
    # Value
    "WAR", "Clutch", "WPA",
    # Context
    "G", "IP", "gmLI",
]

BQ_PROJECT = "data-platform-490901"
BQ_DATASET = "mlb_wp"


def fetch_batting(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch FanGraphs batting stats and map to MLBAM IDs."""
    from pybaseball import batting_stats

    frames = []
    for year in range(start, end + 1):
        print(f"  [{year}] Fetching FanGraphs batting stats...")
        t0 = time.time()

        for attempt in range(1, 4):
            try:
                df = batting_stats(year, year, qual=50)
                break
            except Exception as e:
                print(f"    Attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(10 * attempt)
                else:
                    print(f"    Skipping {year}")
                    df = None

        if df is None or len(df) == 0:
            continue

        elapsed = time.time() - t0

        # Keep IDfg + Name + all features that exist
        keep = ["IDfg", "Name"]
        for feat in BATTER_FEATURES:
            if feat in df.columns:
                keep.append(feat)
        df = df[[c for c in keep if c in df.columns]].copy()
        df["season"] = year

        frames.append(df)
        print(f"    {len(df)} batters, {len(df.columns)} cols in {elapsed:.1f}s")
        time.sleep(2)

    if not frames:
        print("ERROR: No batting data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _map_to_mlbam(combined)

    output_path = output_dir / "fg_batting.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")
    return combined


def fetch_pitching(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch FanGraphs pitching stats and map to MLBAM IDs."""
    from pybaseball import pitching_stats

    frames = []
    for year in range(start, end + 1):
        print(f"  [{year}] Fetching FanGraphs pitching stats...")
        t0 = time.time()

        for attempt in range(1, 4):
            try:
                df = pitching_stats(year, year, qual=30)
                break
            except Exception as e:
                print(f"    Attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(10 * attempt)
                else:
                    print(f"    Skipping {year}")
                    df = None

        if df is None or len(df) == 0:
            continue

        elapsed = time.time() - t0

        keep = ["IDfg", "Name"]
        for feat in PITCHER_FEATURES:
            if feat in df.columns:
                keep.append(feat)
        df = df[[c for c in keep if c in df.columns]].copy()
        df["season"] = year

        frames.append(df)
        print(f"    {len(df)} pitchers, {len(df.columns)} cols in {elapsed:.1f}s")
        time.sleep(2)

    if not frames:
        print("ERROR: No pitching data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _map_to_mlbam(combined)

    output_path = output_dir / "fg_pitching.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")
    return combined


def _map_to_mlbam(df: pd.DataFrame) -> pd.DataFrame:
    """Map FanGraphs IDs to MLBAM IDs via Chadwick register."""
    from pybaseball import chadwick_register

    print("\n  Mapping FanGraphs ID -> MLBAM ID via Chadwick register...")
    reg = chadwick_register()

    # Filter valid entries
    valid = reg[reg["key_fangraphs"].notna() & reg["key_mlbam"].notna()]
    fg_to_mlbam = dict(zip(
        valid["key_fangraphs"].astype(int),
        valid["key_mlbam"].astype(int),
    ))

    df = df.copy()
    df["player_id"] = df["IDfg"].astype(int).map(fg_to_mlbam)
    n_mapped = df["player_id"].notna().sum()
    n_total = len(df)
    print(f"  Mapped: {n_mapped}/{n_total} ({n_mapped / n_total * 100:.1f}%)")

    df = df.dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df = df.drop(columns=["IDfg"], errors="ignore")

    return df


def load_to_bq(df: pd.DataFrame, table_name: str):
    """Load DataFrame to BigQuery."""
    from google.cloud import bigquery

    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    client = bigquery.Client(project=BQ_PROJECT)
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    print(f"BQ: {table_ref} — {table.num_rows:,} rows, {len(table.schema)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch FanGraphs batting/pitching stats -> CSV + BQ")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="Start year (matching Statcast data range)")
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for CSVs")
    parser.add_argument("--no-bq", action="store_true",
                        help="Skip BQ upload (CSV only)")
    parser.add_argument("--batting-only", action="store_true")
    parser.add_argument("--pitching-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bat_df, pit_df = pd.DataFrame(), pd.DataFrame()

    if not args.pitching_only:
        print(f"{'=' * 60}")
        print(f"BATTING STATS ({args.start_year}-{args.end_year})")
        print(f"{'=' * 60}")
        bat_df = fetch_batting(args.start_year, args.end_year, output_dir)

    if not args.batting_only:
        print(f"\n{'=' * 60}")
        print(f"PITCHING STATS ({args.start_year}-{args.end_year})")
        print(f"{'=' * 60}")
        pit_df = fetch_pitching(args.start_year, args.end_year, output_dir)

    # Coverage reports
    for label, df, features in [
        ("BATTING", bat_df, BATTER_FEATURES),
        ("PITCHING", pit_df, PITCHER_FEATURES),
    ]:
        if len(df) == 0:
            continue
        print(f"\n{'=' * 60}")
        print(f"{label} COVERAGE REPORT")
        print(f"{'=' * 60}")
        print(f"Years: {df['season'].nunique()}, Total rows: {len(df):,}")
        print(f"Columns in CSV: {[c for c in df.columns if c not in ('Name', 'player_id', 'season')]}")

        # Check which requested features are missing
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"\n  MISSING features (not in FanGraphs data): {missing}")

        # Null rates per feature per year
        print(f"\n  {'Feature':<15}", end="")
        years = sorted(df["season"].unique())
        for y in years:
            print(f"  {y}", end="")
        print()

        for feat in features:
            if feat in df.columns:
                print(f"  {feat:<15}", end="")
                for y in years:
                    mask = df["season"] == y
                    null_pct = df.loc[mask, feat].isna().mean() * 100
                    print(f"  {null_pct:5.1f}", end="")
                print()

    # BQ upload
    if not args.no_bq:
        if len(bat_df) > 0 and not args.pitching_only:
            print("\nLoading batting stats to BQ...")
            load_to_bq(bat_df, "fg_batting_stats")
        if len(pit_df) > 0 and not args.batting_only:
            print("Loading pitching stats to BQ...")
            load_to_bq(pit_df, "fg_pitching_stats")


if __name__ == "__main__":
    main()
