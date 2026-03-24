"""
Fetch Statcast fielding and baserunning metrics for WP model enrichment.

Downloads sprint speed, outs above average (OAA), catcher framing,
and catcher pop time from Baseball Savant via pybaseball.

These player-season metrics are joined to pitch-level Statcast data
in train_wp_statcast.py to provide defensive quality context.

Usage:
  python scripts/fetch_fielding_running.py --start-year 2016 --end-year 2024
  python scripts/fetch_fielding_running.py --no-bq   # CSV only
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

BQ_PROJECT = "data-platform-490901"
BQ_DATASET = "mlb_wp"

# Sprint speed available from 2015, OAA from 2016, catcher framing from 2015
MIN_YEAR_OAA = 2016
MIN_YEAR_SPRINT = 2015
MIN_YEAR_CATCHER = 2015


def fetch_sprint_speed(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch Statcast sprint speed for all players."""
    from pybaseball import statcast_sprint_speed

    frames = []
    for year in range(max(start, MIN_YEAR_SPRINT), end + 1):
        print(f"  [{year}] Fetching sprint speed...")
        t0 = time.time()

        for attempt in range(1, 4):
            try:
                df = statcast_sprint_speed(year, min_opp=10)
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

        # Standardize column names and add season
        df = df.copy()
        df["season"] = year

        frames.append(df)
        print(f"    {len(df)} players in {elapsed:.1f}s")
        time.sleep(3)

    if not frames:
        print("ERROR: No sprint speed data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Rename key columns for consistency
    # pybaseball returns: player_id (MLBAM), sprint_speed, hp_to_1b, etc.
    # Ensure player_id is int
    if "player_id" in combined.columns:
        combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce")
        combined = combined.dropna(subset=["player_id"])
        combined["player_id"] = combined["player_id"].astype(int)

    output_path = output_dir / "statcast_sprint_speed.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")

    # Coverage report
    print(f"\n  Sprint Speed Coverage:")
    for year in sorted(combined["season"].unique()):
        n = len(combined[combined["season"] == year])
        speed_col = _find_speed_col(combined)
        if speed_col:
            avg = combined.loc[combined["season"] == year, speed_col].mean()
            null_pct = combined.loc[combined["season"] == year, speed_col].isna().mean() * 100
            print(f"    {year}: {n} players, avg={avg:.1f} ft/s, null={null_pct:.1f}%")
        else:
            print(f"    {year}: {n} players")

    return combined


def _find_speed_col(df: pd.DataFrame) -> str | None:
    """Find the sprint speed column (varies by pybaseball version)."""
    candidates = ["sprint_speed", "r_sprint_speed_top50percent", "hp_to_1b"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_oaa(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch OAA for all fielding positions."""
    from pybaseball import statcast_outs_above_average

    # Position codes: 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF
    positions = [3, 4, 5, 6, 7, 8, 9]
    pos_names = {3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}

    frames = []
    for year in range(max(start, MIN_YEAR_OAA), end + 1):
        for pos in positions:
            print(f"  [{year}] OAA for {pos_names[pos]} (pos={pos})...")
            t0 = time.time()

            for attempt in range(1, 4):
                try:
                    df = statcast_outs_above_average(year, pos, min_att="q")
                    break
                except Exception as e:
                    print(f"    Attempt {attempt}/3 failed: {e}")
                    if attempt < 3:
                        time.sleep(10 * attempt)
                    else:
                        print(f"    Skipping {year} {pos_names[pos]}")
                        df = None

            if df is None or len(df) == 0:
                continue

            elapsed = time.time() - t0

            df = df.copy()
            # OAA data has 'year' column from Savant; rename to 'season' for consistency
            if "year" in df.columns:
                df = df.rename(columns={"year": "season"})
            else:
                df["season"] = year
            df["position"] = pos

            frames.append(df)
            print(f"    {len(df)} players in {elapsed:.1f}s")
            time.sleep(2)

    if not frames:
        print("ERROR: No OAA data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Ensure player_id is int
    if "player_id" in combined.columns:
        combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce")
        combined = combined.dropna(subset=["player_id"])
        combined["player_id"] = combined["player_id"].astype(int)

    output_path = output_dir / "statcast_oaa.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")

    # Coverage report
    print(f"\n  OAA Coverage:")
    for year in sorted(combined["season"].unique()):
        n = len(combined[combined["season"] == year])
        oaa_col = _find_oaa_col(combined)
        if oaa_col:
            avg = combined.loc[combined["season"] == year, oaa_col].mean()
            print(f"    {year}: {n} player-positions, avg OAA={avg:.1f}")
        else:
            print(f"    {year}: {n} player-positions")

    # Also create team-level aggregate
    team_oaa = _aggregate_team_oaa(combined)
    if len(team_oaa) > 0:
        team_path = output_dir / "statcast_team_oaa.csv"
        team_oaa.to_csv(team_path, index=False)
        print(f"\n  Team OAA: {team_path} ({len(team_oaa)} team-seasons)")

    return combined


def _find_oaa_col(df: pd.DataFrame) -> str | None:
    """Find the OAA column."""
    candidates = ["outs_above_average", "oaa", "OAA"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _aggregate_team_oaa(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OAA by team + season for team-level fielding quality."""
    oaa_col = _find_oaa_col(df)
    # pybaseball returns 'display_team_name' (e.g. "Nationals", "Brewers")
    team_col = None
    for c in ["display_team_name", "team", "team_name", "team_id"]:
        if c in df.columns:
            team_col = c
            break

    if oaa_col is None or team_col is None:
        print("  WARNING: Cannot aggregate team OAA (missing columns)")
        print(f"    Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Filter out multi-team entries ("---") which can't be mapped
    df_filtered = df[df[team_col] != "---"].copy()
    if len(df_filtered) < len(df):
        print(f"  Filtered {len(df) - len(df_filtered)} multi-team ('---') entries for team aggregation")

    team_oaa = df_filtered.groupby([team_col, "season"]).agg(
        total_oaa=(oaa_col, "sum"),
        avg_oaa=(oaa_col, "mean"),
        n_qualified_fielders=(oaa_col, "count"),
    ).reset_index()

    team_oaa = team_oaa.rename(columns={team_col: "team_name"})

    # Map team display names to abbreviations used in Statcast
    team_oaa["team_abbrev"] = team_oaa["team_name"].map(TEAM_NAME_TO_ABBREV)
    unmapped = team_oaa[team_oaa["team_abbrev"].isna()]["team_name"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: Unmapped team names: {list(unmapped)}")

    return team_oaa


# Map from pybaseball OAA team display names to Statcast abbreviations
TEAM_NAME_TO_ABBREV = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "D-backs": "ARI",
    "Diamondbacks": "ARI", "Dodgers": "LAD", "Giants": "SF",
    "Guardians": "CLE", "Indians": "CLE", "Mariners": "SEA",
    "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI",
    "Pirates": "PIT", "Rangers": "TEX", "Rays": "TB",
    "Red Sox": "BOS", "Reds": "CIN", "Rockies": "COL",
    "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY",
}


def fetch_catcher_stats(start: int, end: int, output_dir: Path) -> pd.DataFrame:
    """Fetch catcher pop time (primary) and framing (best-effort).

    Note: statcast_catcher_framing() has known CSV parsing issues in some
    pybaseball versions. Pop time is the primary data source; framing is
    fetched best-effort and skipped on parse errors.
    """
    from pybaseball import statcast_catcher_poptime

    poptime_frames = []
    framing_frames = []

    for year in range(max(start, MIN_YEAR_CATCHER), end + 1):
        # Pop time (primary — uses entity_id, not player_id)
        print(f"  [{year}] Fetching catcher pop time...")
        for attempt in range(1, 4):
            try:
                pt = statcast_catcher_poptime(year, min_2b_att=5)
                break
            except Exception as e:
                print(f"    Attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(10 * attempt)
                else:
                    pt = None

        if pt is not None and len(pt) > 0:
            pt = pt.copy()
            pt["season"] = year
            # Rename entity_id -> player_id for consistent join key
            if "entity_id" in pt.columns and "player_id" not in pt.columns:
                pt = pt.rename(columns={"entity_id": "player_id"})
            poptime_frames.append(pt)
            print(f"    {len(pt)} catchers (pop time)")
        time.sleep(2)

        # Framing (best-effort — known parsing issues)
        print(f"  [{year}] Fetching catcher framing...")
        try:
            from pybaseball import statcast_catcher_framing
            fr = statcast_catcher_framing(year, min_called_p="q")
            if fr is not None and len(fr) > 0:
                fr = fr.copy()
                fr["season"] = year
                framing_frames.append(fr)
                print(f"    {len(fr)} catchers (framing)")
        except Exception as e:
            print(f"    Framing fetch failed (known issue): {type(e).__name__}: {str(e)[:100]}")
        time.sleep(2)

    # Process poptime
    poptime_df = pd.concat(poptime_frames, ignore_index=True) if poptime_frames else pd.DataFrame()
    framing_df = pd.concat(framing_frames, ignore_index=True) if framing_frames else pd.DataFrame()

    if len(poptime_df) == 0 and len(framing_df) == 0:
        print("ERROR: No catcher data fetched")
        return pd.DataFrame()

    # Ensure player_id is int
    for df in [poptime_df, framing_df]:
        if len(df) > 0 and "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
            df.dropna(subset=["player_id"], inplace=True)
            df["player_id"] = df["player_id"].astype(int)

    # Merge on player_id + season
    if len(poptime_df) > 0 and len(framing_df) > 0:
        # Prefix framing columns to avoid collisions
        fr_rename = {c: f"fr_{c}" for c in framing_df.columns
                     if c not in ("player_id", "season")}
        framing_df = framing_df.rename(columns=fr_rename)
        combined = poptime_df.merge(framing_df, on=["player_id", "season"], how="outer")
    elif len(poptime_df) > 0:
        combined = poptime_df
    else:
        combined = framing_df

    output_path = output_dir / "statcast_catcher.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(combined):,} rows, {len(combined.columns)} cols)")

    # Coverage report
    print(f"\n  Catcher Coverage:")
    for year in sorted(combined["season"].unique()):
        n = len(combined[combined["season"] == year])
        cols_info = []
        for col in ["pop_2b_sba", "maxeff_arm_2b_3b_sba"]:
            if col in combined.columns:
                null_pct = combined.loc[combined["season"] == year, col].isna().mean() * 100
                cols_info.append(f"{col} null={null_pct:.0f}%")
        print(f"    {year}: {n} catchers" + (f" ({', '.join(cols_info)})" if cols_info else ""))

    return combined


def _clean_col_names_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for BQ compatibility."""
    col_map = {}
    for c in df.columns:
        new = c.replace("/", "_").replace("%", "_pct").replace("+", "_plus")
        new = new.replace(" ", "_").replace("-", "_")
        col_map[c] = new
    return df.rename(columns=col_map)


def load_to_bq(df: pd.DataFrame, table_name: str):
    """Load DataFrame to BigQuery."""
    from google.cloud import bigquery

    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    df_bq = _clean_col_names_for_bq(df.copy())

    client = bigquery.Client(project=BQ_PROJECT)
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df_bq, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    print(f"BQ: {table_ref} — {table.num_rows:,} rows, {len(table.schema)} cols")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Statcast fielding/running stats -> CSV + BQ")
    parser.add_argument("--start-year", type=int, default=2016,
                        help="Start year (OAA available from 2016)")
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", default="data/",
                        help="Output directory for CSVs")
    parser.add_argument("--no-bq", action="store_true",
                        help="Skip BQ upload (CSV only)")
    parser.add_argument("--sprint-only", action="store_true")
    parser.add_argument("--oaa-only", action="store_true")
    parser.add_argument("--catcher-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sprint_df = pd.DataFrame()
    oaa_df = pd.DataFrame()
    catcher_df = pd.DataFrame()

    run_all = not (args.sprint_only or args.oaa_only or args.catcher_only)

    if run_all or args.sprint_only:
        print(f"{'=' * 60}")
        print(f"SPRINT SPEED ({args.start_year}-{args.end_year})")
        print(f"{'=' * 60}")
        sprint_df = fetch_sprint_speed(args.start_year, args.end_year, output_dir)

    if run_all or args.oaa_only:
        print(f"\n{'=' * 60}")
        print(f"OUTS ABOVE AVERAGE ({max(args.start_year, MIN_YEAR_OAA)}-{args.end_year})")
        print(f"{'=' * 60}")
        oaa_df = fetch_oaa(args.start_year, args.end_year, output_dir)

    if run_all or args.catcher_only:
        print(f"\n{'=' * 60}")
        print(f"CATCHER STATS ({args.start_year}-{args.end_year})")
        print(f"{'=' * 60}")
        catcher_df = fetch_catcher_stats(args.start_year, args.end_year, output_dir)

    # Column summary
    for label, df in [("Sprint Speed", sprint_df), ("OAA", oaa_df), ("Catcher", catcher_df)]:
        if len(df) > 0:
            print(f"\n{label} columns: {list(df.columns)}")
            null_report = df.isnull().mean() * 100
            high_null = null_report[null_report > 0]
            if len(high_null) > 0:
                print(f"  Columns with nulls:")
                for col, pct in high_null.items():
                    print(f"    {col}: {pct:.1f}%")

    # BQ upload
    if not args.no_bq:
        if len(sprint_df) > 0:
            print("\nLoading sprint speed to BQ...")
            load_to_bq(sprint_df, "statcast_sprint_speed")
        if len(oaa_df) > 0:
            print("Loading OAA to BQ...")
            load_to_bq(oaa_df, "statcast_oaa")
            # Also load team aggregate
            team_oaa = _aggregate_team_oaa(oaa_df)
            if len(team_oaa) > 0:
                print("Loading team OAA to BQ...")
                load_to_bq(team_oaa, "statcast_team_oaa")
        if len(catcher_df) > 0:
            print("Loading catcher stats to BQ...")
            load_to_bq(catcher_df, "statcast_catcher")


if __name__ == "__main__":
    main()
