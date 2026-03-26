"""
Validate FanGraphs feature availability in mlb_shared BQ tables.

Checks:
1. Every feature used by train_wp_statcast.py exists in the BQ table
2. Null rate per feature per year (2015-2025)
3. Flags 100% null years (Stuff+ etc.)
4. Basic statistics per feature

Usage:
  python scripts/validate_fg_features.py
  python scripts/validate_fg_features.py --csv  # validate local CSVs instead of BQ
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Features used by train_wp_statcast.py engineer_features(), after
# mlb-data-pipeline sanitize_columns() rules:
#   % -> _pct, / -> _per_, + -> _plus, trailing - -> _minus

BATTER_FEATURES = [
    "wOBA", "xwOBA", "wRC_plus", "OPS", "AVG", "OBP", "SLG", "ISO", "BABIP",
    "K_pct", "BB_pct", "O_Swing_pct", "Z_Swing_pct", "O_Contact_pct",
    "Z_Contact_pct", "Zone_pct", "SwStr_pct", "Contact_pct",
    "GB_pct", "FB_pct", "LD_pct", "IFFB_pct", "HR_per_FB",
    "Pull_pct", "Cent_pct", "Oppo_pct",
    "Soft_pct", "Med_pct", "Hard_pct", "HardHit_pct",
    "WAR", "Off", "Def", "BsR", "Spd",
    "Clutch", "WPA", "wRAA",
    "wFB_per_C", "wSL_per_C", "wCH_per_C",
    "G", "PA",
]

PITCHER_FEATURES = [
    "ERA", "FIP", "xFIP", "SIERA",
    "ERA_minus", "FIP_minus", "xFIP_minus",
    "K_pct", "BB_pct", "K_BB_pct", "K_per_9", "BB_per_9", "K_per_BB",
    "HR_per_9", "HR_per_FB", "WHIP", "BABIP", "LOB_pct",
    "SwStr_pct", "CSW_pct",
    "O_Swing_pct", "Z_Swing_pct", "O_Contact_pct", "Z_Contact_pct",
    "Zone_pct", "F_Strike_pct",
    "Stuff_plus", "Location_plus", "Pitching_plus",
    "GB_pct", "FB_pct", "LD_pct", "IFFB_pct",
    "Pull_pct", "Cent_pct", "Oppo_pct",
    "Soft_pct", "Med_pct", "Hard_pct",
    "wFB_per_C", "wSL_per_C", "wCH_per_C",
    "GS", "Start_IP", "Relief_IP",
    "WAR", "Clutch", "WPA",
    "G", "IP", "gmLI",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def validate_features(csv_path: Path, feature_list: list[str], label: str):
    """Validate feature list against actual CSV columns."""
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return False

    df = pd.read_csv(csv_path)
    years = sorted(df["season"].unique())
    skip_cols = {"name", "player_id", "season", "fg_id"}
    csv_cols = set(df.columns) - skip_cols

    print(f"\n{'=' * 80}")
    print(f"{label} VALIDATION")
    print(f"{'=' * 80}")
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df):,} | Years: {years[0]}-{years[-1]} | Columns: {len(csv_cols)}")

    # 1. Coverage check: features vs CSV columns
    print(f"\n--- 1. COVERAGE CHECK ---")
    all_present = True
    for feat in feature_list:
        if feat not in df.columns:
            print(f"  MISSING: '{feat}' not in CSV!")
            all_present = False
    if all_present:
        print(f"  OK: All {len(feature_list)} features exist in CSV")

    # Extra columns in CSV not in feature list
    extra = csv_cols - set(feature_list)
    if extra:
        print(f"  Extra CSV columns (not in feature list): {len(extra)} cols")

    # 2. Null rate per feature per year
    print(f"\n--- 2. NULL RATE PER FEATURE PER YEAR (%) ---")
    header = f"  {'Feature':<15}"
    for y in years:
        header += f" {y:>6}"
    header += "   OVERALL"
    print(header)
    print("  " + "-" * (len(header) - 2))

    has_issues = False
    for feat in feature_list:
        if feat not in df.columns:
            continue
        row = f"  {feat:<15}"
        overall_null = df[feat].isna().mean() * 100
        year_nulls = []
        for y in years:
            mask = df["season"] == y
            null_pct = df.loc[mask, feat].isna().mean() * 100
            year_nulls.append(null_pct)
            if null_pct == 100.0:
                row += f"  {'N/A':>5}"
            elif null_pct > 0:
                row += f" {null_pct:5.1f}%"
            else:
                row += f"   {'0.0':>4}"
        row += f"   {overall_null:5.1f}%"

        if any(n == 100.0 for n in year_nulls):
            row += "  <-- 100% null in some years"
            has_issues = True
        print(row)

    if not has_issues:
        print(f"\n  No features with 100% null years.")

    # 3. Basic statistics
    print(f"\n--- 3. BASIC STATISTICS ---")
    print(f"  {'Feature':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Rows':>6}")
    print("  " + "-" * 60)
    for feat in feature_list:
        if feat not in df.columns:
            continue
        s = df[feat]
        valid = s.dropna()
        print(f"  {feat:<15} {valid.mean():>8.2f} {valid.std():>8.2f} "
              f"{valid.min():>8.2f} {valid.max():>8.2f} {len(valid):>6}")

    # 4. Per-year row counts
    print(f"\n--- 4. PLAYER COUNT PER YEAR ---")
    for y in years:
        n = (df["season"] == y).sum()
        print(f"  {y}: {n:>4} players")

    return all_present


def main():
    print("FanGraphs Feature Validation (mlb_shared pipeline data)")
    print("=" * 80)

    bat_ok = validate_features(
        DATA_DIR / "fg_batting.csv", BATTER_FEATURES, "BATTING")
    pit_ok = validate_features(
        DATA_DIR / "fg_pitching.csv", PITCHER_FEATURES, "PITCHING")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Batting features:  {len(BATTER_FEATURES)} defined, "
          f"all present = {bat_ok}")
    print(f"  Pitching features: {len(PITCHER_FEATURES)} defined, "
          f"all present = {pit_ok}")
    print()

    print("  NOTES:")
    print("  - Stuff+/Location+/Pitching+ are available 2020+ only (100% null 2015-2019)")
    print("  - LightGBM/CatBoost handle NaN natively (no imputation needed)")
    print("  - Data sourced from mlb_shared dataset (mlb-data-pipeline)")

    if bat_ok and pit_ok:
        print("\n  VALIDATION PASSED")
        return 0
    else:
        print("\n  VALIDATION FAILED - check missing features above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
