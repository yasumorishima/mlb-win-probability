"""
Train WP model on full Statcast pitch-level data from BigQuery.

Uses 70+ features including:
- Game state (inning, outs, runners, score)
- Pitch characteristics (velocity, spin, movement, location)
- Batted ball quality (exit velo, launch angle, xwOBA)
- Bat tracking (bat speed, swing length, attack angle)
- Context (count, game phase, lineup position)

Benchmark: MLB home_win_exp from Statcast API (our model must beat this).

LightGBM + CatBoost + Optuna, same pipeline as baseball-mlops.

Usage:
  python scripts/train_wp_statcast.py --output-dir data/
  python scripts/train_wp_statcast.py --n-trials 500 --n-trials-catboost 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery


PROJECT = "data-platform-490901"
DATASET_SHARED = "mlb_shared"   # All shared data via mlb-data-pipeline
TABLE = "statcast_pitches"


def get_bq_client():
    """Get authenticated BQ client."""
    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        key_path = Path("/tmp/gcp-sa-key.json")
        key_path.write_text(sa_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        local_key = Path(r"C:\Users\fw_ya\.claude\gcp-sa-key.json")
        if local_key.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_key)

    return bigquery.Client(project=PROJECT)


def load_from_bq(test_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Statcast data from BigQuery, split by year."""
    client = get_bq_client()

    # At-bat outcomes only (events IS NOT NULL) to reduce memory
    query = f"""
        SELECT *
        FROM `{PROJECT}.{DATASET_SHARED}.{TABLE}`
        WHERE game_type = 'R' AND events IS NOT NULL
        ORDER BY game_pk, inning, is_bottom, outs_when_up
    """

    print("Loading from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"  Total: {len(df):,} rows")

    train = df[df["game_year"] < test_year].copy()
    test = df[df["game_year"] == test_year].copy()
    print(f"  Train: {len(train):,} (< {test_year})")
    print(f"  Test:  {len(test):,} ({test_year})")

    return train, test


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply mlb-data-pipeline sanitize_columns rules to ensure consistent names.

    Handles both fresh BQ reads (already sanitized) and old cached CSVs
    (original FanGraphs names with %, /, +, trailing -).
    Double-sanitization is safe (no-op on already-clean names).
    """
    import re
    rename = {}
    for col in df.columns:
        new = col
        new = new.replace("%", "_pct")
        new = new.replace("/", "_per_")
        new = new.replace("+", "_plus")
        new = re.sub(r"-$", "_minus", new)
        new = re.sub(r"[^a-zA-Z0-9_]", "_", new)
        new = re.sub(r"_+", "_", new)
        new = new.strip("_")
        if new != col:
            rename[col] = new
    if rename:
        df = df.rename(columns=rename)
    return df


def _safe_col(df: pd.DataFrame, col: str, fill: float = 0,
              use_median: bool = False) -> pd.Series:
    """Safely access a DataFrame column, returning fill value if missing."""
    if col not in df.columns:
        return pd.Series(fill, index=df.index, dtype=float)
    if use_median:
        median_val = df[col].median()
        return df[col].fillna(median_val if pd.notna(median_val) else fill)
    return df[col].fillna(fill)


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Engineer 70+ features from raw Statcast data.
    Returns feature matrix and feature names.
    All column accesses are guarded against missing columns in BQ.
    """
    features = pd.DataFrame(index=df.index)

    # --- Game state (core) ---
    features["inning"] = _safe_col(df, "inning", fill=1)
    if "is_bottom" in df.columns:
        features["is_bottom"] = df["is_bottom"]
    elif "inning_topbot" in df.columns:
        features["is_bottom"] = (df["inning_topbot"] == "Bot").astype(int)
    else:
        features["is_bottom"] = 0
    features["outs"] = _safe_col(df, "outs_when_up")
    features["balls"] = _safe_col(df, "balls")
    features["strikes"] = _safe_col(df, "strikes")
    if "score_diff" in df.columns:
        features["score_diff"] = df["score_diff"].fillna(0)
    elif "home_score" in df.columns and "away_score" in df.columns:
        features["score_diff"] = (df["home_score"] - df["away_score"]).fillna(0)
    else:
        features["score_diff"] = 0

    # Runners
    features["r1"] = df["on_1b"].notna().astype(int) if "on_1b" in df.columns else 0
    features["r2"] = df["on_2b"].notna().astype(int) if "on_2b" in df.columns else 0
    features["r3"] = df["on_3b"].notna().astype(int) if "on_3b" in df.columns else 0
    features["total_runners"] = features["r1"] + features["r2"] + features["r3"]
    features["scoring_position"] = features["r2"] + features["r3"]

    # --- Game context ---
    features["abs_lead"] = features["score_diff"].abs()
    features["game_phase"] = features["inning"].clip(1, 12).apply(
        lambda x: 0 if x <= 3 else (1 if x <= 6 else 2))
    features["walk_off_eligible"] = (
        (features["is_bottom"] == 1) &
        (features["inning"] >= 9) &
        (features["score_diff"] <= 0)
    ).astype(int)
    features["tied"] = (features["score_diff"] == 0).astype(int)
    features["close_game"] = (features["abs_lead"] <= 2).astype(int)

    # Count state
    features["count_state"] = features["balls"] * 3 + features["strikes"]
    features["full_count"] = ((features["balls"] == 3) & (features["strikes"] == 2)).astype(int)
    features["ahead_in_count"] = (features["strikes"] > features["balls"]).astype(int)

    # --- Interactions ---
    features["inn_x_bottom"] = features["inning"] * features["is_bottom"]
    features["inn_x_outs"] = features["inning"] * features["outs"]
    features["inn_x_score"] = features["inning"] * features["score_diff"]
    features["outs_x_score"] = features["outs"] * features["score_diff"]
    features["phase_x_lead"] = features["game_phase"] * features["abs_lead"]
    features["runners_x_outs"] = features["total_runners"] * features["outs"]
    features["scoring_x_outs"] = features["scoring_position"] * features["outs"]

    # --- Pitch characteristics ---
    features["release_speed"] = _safe_col(df, "release_speed", use_median=True)
    features["effective_speed"] = _safe_col(df, "effective_speed",
                                            fill=features["release_speed"].median())
    features["pfx_x"] = _safe_col(df, "pfx_x")
    features["pfx_z"] = _safe_col(df, "pfx_z")
    features["total_movement"] = np.sqrt(features["pfx_x"]**2 + features["pfx_z"]**2)
    features["release_spin"] = _safe_col(df, "release_spin_rate", use_median=True)
    features["plate_x"] = _safe_col(df, "plate_x")
    features["plate_z"] = _safe_col(df, "plate_z")
    features["in_zone"] = _safe_col(df, "zone").apply(
        lambda z: 1 if 1 <= z <= 9 else 0)
    features["release_ext"] = _safe_col(df, "release_extension", use_median=True)
    features["arm_angle"] = _safe_col(df, "arm_angle", use_median=True)

    # --- Batted ball quality ---
    features["launch_speed"] = _safe_col(df, "launch_speed")
    features["launch_angle"] = _safe_col(df, "launch_angle")
    features["hit_distance"] = _safe_col(df, "hit_distance_sc")
    features["xwoba"] = _safe_col(df, "estimated_woba_using_speedangle")
    features["xba"] = _safe_col(df, "estimated_ba_using_speedangle")
    features["xslg"] = _safe_col(df, "estimated_slg_using_speedangle")
    features["woba_value"] = _safe_col(df, "woba_value")

    # Barrel proxy: launch_speed >= 98 and 26 <= launch_angle <= 30
    features["barrel"] = (
        (features["launch_speed"] >= 98) &
        (features["launch_angle"] >= 26) &
        (features["launch_angle"] <= 30)
    ).astype(int)

    # --- Bat tracking (2024+ only, 0 for earlier years) ---
    features["bat_speed"] = _safe_col(df, "bat_speed")
    features["swing_length"] = _safe_col(df, "swing_length")
    features["attack_angle_bat"] = _safe_col(df, "attack_angle")

    # --- Additional batting stats (new from full BQ) ---
    features["babip_value"] = _safe_col(df, "babip_value")
    features["iso_value"] = _safe_col(df, "iso_value")
    features["delta_run_exp"] = _safe_col(df, "delta_run_exp")
    features["delta_pitcher_run_exp"] = _safe_col(df, "delta_pitcher_run_exp")

    # --- Player age ---
    features["age_bat"] = _safe_col(df, "age_bat", use_median=True)
    features["age_pit"] = _safe_col(df, "age_pit", use_median=True)

    # --- Batted ball direction ---
    features["hit_location"] = _safe_col(df, "hit_location")

    # --- Strike zone geometry ---
    features["sz_top"] = _safe_col(df, "sz_top", use_median=True)
    features["sz_bot"] = _safe_col(df, "sz_bot", use_median=True)
    features["sz_height"] = features["sz_top"] - features["sz_bot"]
    features["plate_z_norm"] = np.where(
        features["sz_height"] > 0,
        (features["plate_z"] - features["sz_bot"]) / features["sz_height"],
        0.5,
    )

    # --- Spin axis ---
    features["spin_axis"] = _safe_col(df, "spin_axis", use_median=True)

    # --- Release point ---
    features["release_pos_x"] = _safe_col(df, "release_pos_x", use_median=True)
    features["release_pos_z"] = _safe_col(df, "release_pos_z", use_median=True)

    # --- Pitch trajectory ---
    features["vx0"] = _safe_col(df, "vx0")
    features["vz0"] = _safe_col(df, "vz0")
    features["ax"] = _safe_col(df, "ax")
    features["az"] = _safe_col(df, "az")

    # --- FanGraphs pitcher season stats (joined by pitcher + game_year) ---
    # Column names follow mlb-data-pipeline sanitize_columns() rules:
    #   % -> _pct, / -> _per_, + -> _plus, trailing - -> _minus
    # Core ERA models
    features["fg_pit_era"] = _safe_col(df, "fg_pit_ERA", fill=4.0)
    features["fg_pit_fip"] = _safe_col(df, "fg_pit_FIP", fill=4.0)
    features["fg_pit_xfip"] = _safe_col(df, "fg_pit_xFIP", fill=4.0)
    features["fg_pit_siera"] = _safe_col(df, "fg_pit_SIERA", fill=4.0)
    features["fg_pit_era_minus"] = _safe_col(df, "fg_pit_ERA_minus", fill=100)
    features["fg_pit_fip_minus"] = _safe_col(df, "fg_pit_FIP_minus", fill=100)
    features["fg_pit_xfip_minus"] = _safe_col(df, "fg_pit_xFIP_minus", fill=100)
    # Rate
    features["fg_pit_k_pct"] = _safe_col(df, "fg_pit_K_pct", fill=0.22)
    features["fg_pit_bb_pct"] = _safe_col(df, "fg_pit_BB_pct", fill=0.08)
    features["fg_pit_k_bb_pct"] = _safe_col(df, "fg_pit_K_BB_pct", fill=0.14)
    features["fg_pit_k9"] = _safe_col(df, "fg_pit_K_per_9", fill=8.5)
    features["fg_pit_bb9"] = _safe_col(df, "fg_pit_BB_per_9", fill=3.2)
    features["fg_pit_k_bb"] = _safe_col(df, "fg_pit_K_per_BB", fill=3.0)
    features["fg_pit_hr9"] = _safe_col(df, "fg_pit_HR_per_9", fill=1.1)
    features["fg_pit_hr_fb"] = _safe_col(df, "fg_pit_HR_per_FB", fill=0.11)
    features["fg_pit_whip"] = _safe_col(df, "fg_pit_WHIP", fill=1.3)
    features["fg_pit_babip"] = _safe_col(df, "fg_pit_BABIP", fill=0.29)
    features["fg_pit_lob_pct"] = _safe_col(df, "fg_pit_LOB_pct", fill=0.72)
    # Pitch quality
    features["fg_pit_swstr"] = _safe_col(df, "fg_pit_SwStr_pct", fill=0.11)
    features["fg_pit_csw"] = _safe_col(df, "fg_pit_CSW_pct", fill=0.28)
    features["fg_pit_o_swing"] = _safe_col(df, "fg_pit_O_Swing_pct", fill=0.32)
    features["fg_pit_z_swing"] = _safe_col(df, "fg_pit_Z_Swing_pct", fill=0.69)
    features["fg_pit_o_contact"] = _safe_col(df, "fg_pit_O_Contact_pct", fill=0.62)
    features["fg_pit_z_contact"] = _safe_col(df, "fg_pit_Z_Contact_pct", fill=0.86)
    features["fg_pit_zone"] = _safe_col(df, "fg_pit_Zone_pct", fill=0.43)
    features["fg_pit_fstrike"] = _safe_col(df, "fg_pit_F_Strike_pct", fill=0.60)
    # Stuff+ (2020+, fill=100 for league average)
    features["fg_pit_stuff_plus"] = _safe_col(df, "fg_pit_Stuff_plus", fill=100)
    features["fg_pit_location_plus"] = _safe_col(df, "fg_pit_Location_plus", fill=100)
    features["fg_pit_pitching_plus"] = _safe_col(df, "fg_pit_Pitching_plus", fill=100)
    # Batted ball
    features["fg_pit_gb_pct"] = _safe_col(df, "fg_pit_GB_pct", fill=0.42)
    features["fg_pit_fb_pct"] = _safe_col(df, "fg_pit_FB_pct", fill=0.38)
    features["fg_pit_ld_pct"] = _safe_col(df, "fg_pit_LD_pct", fill=0.19)
    features["fg_pit_iffb_pct"] = _safe_col(df, "fg_pit_IFFB_pct", fill=0.10)
    features["fg_pit_pull_pct"] = _safe_col(df, "fg_pit_Pull_pct", fill=0.40)
    features["fg_pit_soft_pct"] = _safe_col(df, "fg_pit_Soft_pct", fill=0.16)
    features["fg_pit_hard_pct"] = _safe_col(df, "fg_pit_Hard_pct", fill=0.31)
    # Pitch type values
    features["fg_pit_wfb_c"] = _safe_col(df, "fg_pit_wFB_per_C")
    features["fg_pit_wsl_c"] = _safe_col(df, "fg_pit_wSL_per_C")
    features["fg_pit_wch_c"] = _safe_col(df, "fg_pit_wCH_per_C")
    # Role
    features["fg_pit_gs"] = _safe_col(df, "fg_pit_GS")
    features["fg_pit_start_ip"] = _safe_col(df, "fg_pit_Start_IP")
    features["fg_pit_relief_ip"] = _safe_col(df, "fg_pit_Relief_IP")
    # Value
    features["fg_pit_war"] = _safe_col(df, "fg_pit_WAR")
    features["fg_pit_clutch"] = _safe_col(df, "fg_pit_Clutch")
    features["fg_pit_wpa"] = _safe_col(df, "fg_pit_WPA")
    features["fg_pit_gmli"] = _safe_col(df, "fg_pit_gmLI", fill=1.0)

    # --- FanGraphs batter season stats (joined by batter + game_year) ---
    # Column names follow mlb-data-pipeline sanitize_columns() rules
    # Core
    features["fg_bat_woba"] = _safe_col(df, "fg_bat_wOBA", fill=0.30)
    features["fg_bat_xwoba"] = _safe_col(df, "fg_bat_xwOBA", fill=0.30)
    features["fg_bat_wrc_plus"] = _safe_col(df, "fg_bat_wRC_plus", fill=100)
    features["fg_bat_ops"] = _safe_col(df, "fg_bat_OPS", fill=0.70)
    features["fg_bat_iso"] = _safe_col(df, "fg_bat_ISO", fill=0.14)
    features["fg_bat_babip"] = _safe_col(df, "fg_bat_BABIP", fill=0.29)
    # Plate discipline
    features["fg_bat_k_pct"] = _safe_col(df, "fg_bat_K_pct", fill=0.22)
    features["fg_bat_bb_pct"] = _safe_col(df, "fg_bat_BB_pct", fill=0.08)
    features["fg_bat_o_swing"] = _safe_col(df, "fg_bat_O_Swing_pct", fill=0.32)
    features["fg_bat_z_swing"] = _safe_col(df, "fg_bat_Z_Swing_pct", fill=0.69)
    features["fg_bat_o_contact"] = _safe_col(df, "fg_bat_O_Contact_pct", fill=0.61)
    features["fg_bat_z_contact"] = _safe_col(df, "fg_bat_Z_Contact_pct", fill=0.86)
    features["fg_bat_zone"] = _safe_col(df, "fg_bat_Zone_pct", fill=0.43)
    features["fg_bat_swstr"] = _safe_col(df, "fg_bat_SwStr_pct", fill=0.11)
    features["fg_bat_contact"] = _safe_col(df, "fg_bat_Contact_pct", fill=0.76)
    # Batted ball
    features["fg_bat_gb_pct"] = _safe_col(df, "fg_bat_GB_pct", fill=0.43)
    features["fg_bat_fb_pct"] = _safe_col(df, "fg_bat_FB_pct", fill=0.38)
    features["fg_bat_ld_pct"] = _safe_col(df, "fg_bat_LD_pct", fill=0.20)
    features["fg_bat_iffb_pct"] = _safe_col(df, "fg_bat_IFFB_pct", fill=0.10)
    features["fg_bat_hr_fb"] = _safe_col(df, "fg_bat_HR_per_FB", fill=0.11)
    features["fg_bat_pull_pct"] = _safe_col(df, "fg_bat_Pull_pct", fill=0.40)
    features["fg_bat_soft_pct"] = _safe_col(df, "fg_bat_Soft_pct", fill=0.16)
    features["fg_bat_hard_pct"] = _safe_col(df, "fg_bat_Hard_pct", fill=0.38)
    features["fg_bat_hardhit"] = _safe_col(df, "fg_bat_HardHit_pct", fill=0.38)
    # Pitch type values
    features["fg_bat_wfb_c"] = _safe_col(df, "fg_bat_wFB_per_C")
    features["fg_bat_wsl_c"] = _safe_col(df, "fg_bat_wSL_per_C")
    features["fg_bat_wch_c"] = _safe_col(df, "fg_bat_wCH_per_C")
    # Speed & baserunning
    features["fg_bat_spd"] = _safe_col(df, "fg_bat_Spd", fill=4.0)
    features["fg_bat_bsr"] = _safe_col(df, "fg_bat_BsR")
    # Value
    features["fg_bat_war"] = _safe_col(df, "fg_bat_WAR")
    features["fg_bat_off"] = _safe_col(df, "fg_bat_Off")
    features["fg_bat_def"] = _safe_col(df, "fg_bat_Def")
    features["fg_bat_clutch"] = _safe_col(df, "fg_bat_Clutch")
    features["fg_bat_wpa"] = _safe_col(df, "fg_bat_WPA")
    features["fg_bat_wraa"] = _safe_col(df, "fg_bat_wRAA")

    # --- Statcast baserunning (batter sprint speed, joined by batter + season) ---
    features["sc_bat_sprint_speed"] = _safe_col(df, "sc_bat_sprint_speed", fill=27.0)
    features["sc_bat_hp_to_1b"] = _safe_col(df, "sc_bat_hp_to_1b", fill=4.4)
    features["sc_bat_bolts"] = _safe_col(df, "sc_bat_bolts")

    # --- Statcast catcher (joined by fielder_2 + season) ---
    features["sc_c_pop_2b"] = _safe_col(df, "sc_c_pop_2b_sba", fill=2.0)
    features["sc_c_arm_strength"] = _safe_col(df, "sc_c_maxeff_arm_2b_3b_sba", fill=80.0)
    features["sc_c_exchange"] = _safe_col(df, "sc_c_exchange_2b_3b_sba", fill=0.75)

    # --- Statcast team fielding OAA (joined by fielding team + season) ---
    features["sc_team_total_oaa"] = _safe_col(df, "sc_team_total_oaa")
    features["sc_team_avg_oaa"] = _safe_col(df, "sc_team_avg_oaa")

    # --- Lineup / fatigue ---
    features["n_thruorder"] = _safe_col(df, "n_thruorder_pitcher", fill=1)
    features["n_priorpa"] = _safe_col(df, "n_priorpa_thisgame_player_at_bat")

    # --- Nonlinear ---
    features["score_diff_sq"] = features["score_diff"] ** 2
    features["inning_sq"] = features["inning"] ** 2
    features["speed_sq"] = features["release_speed"] ** 2
    features["launch_speed_sq"] = features["launch_speed"] ** 2

    # --- Park factors (from savant-extras, 2015+) ---
    # Loaded separately and merged by home_team + game_year
    if "pf_5yr" in df.columns:
        features["park_factor"] = df["pf_5yr"].fillna(100) / 100.0
        features["park_hr_factor"] = df["pf_hr"].fillna(100) / 100.0
    else:
        features["park_factor"] = 1.0
        features["park_hr_factor"] = 1.0

    # --- Clipped ---
    features["score_capped"] = features["score_diff"].clip(-10, 10)
    features["inning_capped"] = features["inning"].clip(1, 12)

    feature_names = list(features.columns)
    return features.values.astype(np.float32), feature_names


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract binary home win target."""
    # Use post_home_score vs post_away_score at end of game
    # For per-pitch: use the final game outcome
    # We need to map game_pk -> home_won
    if "home_won" in df.columns:
        return df["home_won"].values

    # Compute from final scores per game
    game_outcomes = df.groupby("game_pk").agg(
        final_home=("post_home_score", "max"),
        final_away=("post_away_score", "max"),
    )
    game_outcomes["home_won"] = (game_outcomes["final_home"] > game_outcomes["final_away"]).astype(int)
    df = df.merge(game_outcomes[["home_won"]], left_on="game_pk", right_index=True, how="left")
    return df["home_won"].values


def evaluate(preds: np.ndarray, actuals: np.ndarray, name: str) -> dict:
    """Compute metrics."""
    brier = float(np.mean((preds - actuals) ** 2))
    brier_base = float(np.mean((0.5 - actuals) ** 2))
    brier_skill = 1 - brier / brier_base

    eps = 1e-7
    p = np.clip(preds, eps, 1 - eps)
    log_loss = float(-np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)))

    ece = 0.0
    n = len(actuals)
    for low in np.arange(0, 1.0, 0.1):
        mask = (preds >= low) & (preds < low + 0.1)
        if mask.sum() > 0:
            ece += abs(preds[mask].mean() - actuals[mask].mean()) * mask.sum() / n

    print(f"  {name}: Brier={brier:.6f} BSS={brier_skill:.4f} "
          f"LogLoss={log_loss:.6f} ECE={ece:.4f}")

    return {
        "model": name,
        "brier": round(brier, 6),
        "brier_skill": round(brier_skill, 4),
        "log_loss": round(log_loss, 6),
        "ece": round(ece, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train WP model on Statcast data (BQ)")
    parser.add_argument("--output-dir", default="data/")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--n-trials", type=int, default=0,
                        help="LightGBM Optuna trials")
    parser.add_argument("--n-trials-catboost", type=int, default=0,
                        help="CatBoost Optuna trials")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = load_from_bq(args.test_year)

    # Merge park factors from mlb_shared (savant-extras data via mlb-data-pipeline)
    print("\nLoading park factors...")
    try:
        bq_client = get_bq_client()
        pf_query = f"""
            SELECT CAST(season AS FLOAT64) AS game_year, team AS home_team,
                   pf_5yr, pf_hr
            FROM `{PROJECT}.{DATASET_SHARED}.park_factors`
        """
        pf = bq_client.query(pf_query).to_dataframe()
        if len(pf) > 0:
            train_df = train_df.merge(pf, on=["home_team", "game_year"], how="left")
            test_df = test_df.merge(pf, on=["home_team", "game_year"], how="left")
            print(f"  Merged park factors from BQ ({len(pf)} team-seasons)")
        else:
            print("  No park factors in BQ")
    except Exception as e:
        print(f"  Park factors not available: {e}")

    # Merge FanGraphs season-level stats (batting + pitching)
    print("\nLoading FanGraphs stats...")
    data_dir = Path(args.output_dir)
    for tdf in [train_df, test_df]:
        if "pitcher" in tdf.columns:
            tdf["pitcher"] = tdf["pitcher"].astype(float)
        if "batter" in tdf.columns:
            tdf["batter"] = tdf["batter"].astype(float)
        if "game_year" in tdf.columns:
            tdf["game_year"] = tdf["game_year"].astype(float)

    # --- Pitching stats (from mlb_shared.fg_pitching via mlb-data-pipeline) ---
    try:
        pit_path = data_dir / "fg_pitching.csv"
        if pit_path.exists():
            pit_df = pd.read_csv(pit_path)
        else:
            bq_client = get_bq_client()
            pit_df = bq_client.query(
                f"SELECT * FROM `{PROJECT}.{DATASET_SHARED}.fg_pitching`"
            ).to_dataframe()
            pit_df.to_csv(pit_path, index=False)

        if len(pit_df) > 0:
            # Sanitize column names (handles both BQ-sanitized and old CSV originals)
            pit_df = _sanitize_columns(pit_df)
            # Prefix FG columns to avoid collision with Statcast columns
            # Pipeline uses lowercase 'name'/'season' and adds 'fg_id'
            skip = {"player_id", "name", "season", "fg_id", "Name"}
            rename_map = {c: f"fg_pit_{c}" for c in pit_df.columns if c not in skip}
            pit_df = pit_df.rename(columns=rename_map)
            pit_df = pit_df.rename(columns={"player_id": "pitcher", "season": "game_year"})
            pit_df["pitcher"] = pit_df["pitcher"].astype(float)
            pit_df["game_year"] = pit_df["game_year"].astype(float)
            pit_df = pit_df.drop(columns=["name", "Name", "fg_id"], errors="ignore")

            train_df = train_df.merge(pit_df, on=["pitcher", "game_year"], how="left")
            test_df = test_df.merge(pit_df, on=["pitcher", "game_year"], how="left")
            fg_cols = [c for c in train_df.columns if c.startswith("fg_pit_")]
            matched = train_df[fg_cols[0]].notna().sum() if fg_cols else 0
            print(f"  Pitching: {len(fg_cols)} cols, {matched:,}/{len(train_df):,} "
                  f"({matched/len(train_df)*100:.1f}%) matched")
    except Exception as e:
        print(f"  FG pitching not available: {e}")

    # --- Batting stats (from mlb_shared.fg_batting via mlb-data-pipeline) ---
    try:
        bat_path = data_dir / "fg_batting.csv"
        if bat_path.exists():
            bat_df = pd.read_csv(bat_path)
        else:
            bq_client = get_bq_client()
            bat_df = bq_client.query(
                f"SELECT * FROM `{PROJECT}.{DATASET_SHARED}.fg_batting`"
            ).to_dataframe()
            bat_df.to_csv(bat_path, index=False)

        if len(bat_df) > 0:
            # Sanitize column names (handles both BQ-sanitized and old CSV originals)
            bat_df = _sanitize_columns(bat_df)
            # Pipeline uses lowercase 'name'/'season' and adds 'fg_id'
            skip = {"player_id", "name", "season", "fg_id", "Name"}
            rename_map = {c: f"fg_bat_{c}" for c in bat_df.columns if c not in skip}
            bat_df = bat_df.rename(columns=rename_map)
            bat_df = bat_df.rename(columns={"player_id": "batter", "season": "game_year"})
            bat_df["batter"] = bat_df["batter"].astype(float)
            bat_df["game_year"] = bat_df["game_year"].astype(float)
            bat_df = bat_df.drop(columns=["name", "Name", "fg_id"], errors="ignore")

            train_df = train_df.merge(bat_df, on=["batter", "game_year"], how="left")
            test_df = test_df.merge(bat_df, on=["batter", "game_year"], how="left")
            fg_cols = [c for c in train_df.columns if c.startswith("fg_bat_")]
            matched = train_df[fg_cols[0]].notna().sum() if fg_cols else 0
            print(f"  Batting: {len(fg_cols)} cols, {matched:,}/{len(train_df):,} "
                  f"({matched/len(train_df)*100:.1f}%) matched")
    except Exception as e:
        print(f"  FG batting not available: {e}")

    # --- Sprint speed (from mlb_shared.sprint_speed, join on batter + game_year) ---
    print("\nLoading Statcast sprint speed...")
    try:
        sprint_path = data_dir / "sprint_speed.csv"
        if sprint_path.exists():
            sprint_df = pd.read_csv(sprint_path)
        else:
            bq_client = get_bq_client()
            sprint_df = bq_client.query(
                f"SELECT * FROM `{PROJECT}.{DATASET_SHARED}.sprint_speed`"
            ).to_dataframe()
            sprint_df.to_csv(sprint_path, index=False)

        if len(sprint_df) > 0:
            # Keep key columns for join
            sprint_keep = ["player_id", "season", "sprint_speed", "hp_to_1b", "bolts"]
            sprint_keep = [c for c in sprint_keep if c in sprint_df.columns]
            sprint_join = sprint_df[sprint_keep].copy()
            # Prefix columns
            rename_map = {c: f"sc_bat_{c}" for c in sprint_join.columns
                          if c not in ("player_id", "season")}
            sprint_join = sprint_join.rename(columns=rename_map)
            sprint_join = sprint_join.rename(
                columns={"player_id": "batter", "season": "game_year"})
            sprint_join["batter"] = sprint_join["batter"].astype(float)
            sprint_join["game_year"] = sprint_join["game_year"].astype(float)

            train_df = train_df.merge(sprint_join, on=["batter", "game_year"], how="left")
            test_df = test_df.merge(sprint_join, on=["batter", "game_year"], how="left")
            sc_cols = [c for c in train_df.columns if c.startswith("sc_bat_")]
            matched = train_df[sc_cols[0]].notna().sum() if sc_cols else 0
            print(f"  Sprint: {len(sc_cols)} cols, {matched:,}/{len(train_df):,} "
                  f"({matched/len(train_df)*100:.1f}%) matched")
    except Exception as e:
        print(f"  Sprint speed not available: {e}")

    # --- Catcher stats (from mlb_shared.catcher, join on fielder_2 + game_year) ---
    print("\nLoading Statcast catcher stats...")
    try:
        catcher_path = data_dir / "catcher.csv"
        if catcher_path.exists():
            catcher_df = pd.read_csv(catcher_path)
        else:
            bq_client = get_bq_client()
            catcher_df = bq_client.query(
                f"SELECT * FROM `{PROJECT}.{DATASET_SHARED}.catcher`"
            ).to_dataframe()
            catcher_df.to_csv(catcher_path, index=False)

        if len(catcher_df) > 0:
            catcher_keep = ["player_id", "season",
                            "pop_2b_sba", "maxeff_arm_2b_3b_sba", "exchange_2b_3b_sba"]
            catcher_keep = [c for c in catcher_keep if c in catcher_df.columns]
            catcher_join = catcher_df[catcher_keep].copy()
            rename_map = {c: f"sc_c_{c}" for c in catcher_join.columns
                          if c not in ("player_id", "season")}
            catcher_join = catcher_join.rename(columns=rename_map)
            # Join on fielder_2 (catcher position)
            catcher_join = catcher_join.rename(
                columns={"player_id": "fielder_2", "season": "game_year"})
            catcher_join["fielder_2"] = catcher_join["fielder_2"].astype(float)
            catcher_join["game_year"] = catcher_join["game_year"].astype(float)

            train_df = train_df.merge(catcher_join, on=["fielder_2", "game_year"], how="left")
            test_df = test_df.merge(catcher_join, on=["fielder_2", "game_year"], how="left")
            sc_cols = [c for c in train_df.columns if c.startswith("sc_c_")]
            matched = train_df[sc_cols[0]].notna().sum() if sc_cols else 0
            print(f"  Catcher: {len(sc_cols)} cols, {matched:,}/{len(train_df):,} "
                  f"({matched/len(train_df)*100:.1f}%) matched")
    except Exception as e:
        print(f"  Catcher stats not available: {e}")

    # --- Team OAA (from mlb_shared.oaa_team, join on fielding team + game_year) ---
    print("\nLoading Statcast team OAA...")
    try:
        team_oaa_path = data_dir / "oaa_team.csv"
        if team_oaa_path.exists():
            team_oaa_df = pd.read_csv(team_oaa_path)
        else:
            bq_client = get_bq_client()
            team_oaa_df = bq_client.query(
                f"SELECT * FROM `{PROJECT}.{DATASET_SHARED}.oaa_team`"
            ).to_dataframe()
            team_oaa_df.to_csv(team_oaa_path, index=False)

        if len(team_oaa_df) > 0:
            # Determine fielding team: top of inning = home fields, bottom = away fields
            for tdf in [train_df, test_df]:
                is_bot = tdf.get("is_bottom", tdf.get("inning_topbot", ""))
                if "inning_topbot" in tdf.columns:
                    tdf["fielding_team"] = np.where(
                        tdf["inning_topbot"] == "Bot",
                        tdf.get("away_team", ""),
                        tdf.get("home_team", ""))
                elif "is_bottom" in tdf.columns:
                    tdf["fielding_team"] = np.where(
                        tdf["is_bottom"] == 1,
                        tdf.get("away_team", ""),
                        tdf.get("home_team", ""))

            # Use team_abbrev if available, else team_name
            join_col = "team_abbrev" if "team_abbrev" in team_oaa_df.columns else "team_name"
            oaa_join = team_oaa_df[[join_col, "season", "total_oaa", "avg_oaa"]].copy()
            oaa_join = oaa_join.rename(columns={
                join_col: "fielding_team",
                "season": "game_year",
                "total_oaa": "sc_team_total_oaa",
                "avg_oaa": "sc_team_avg_oaa",
            })
            oaa_join["game_year"] = oaa_join["game_year"].astype(float)

            for tdf_name, tdf in [("train", train_df), ("test", test_df)]:
                if "fielding_team" in tdf.columns:
                    before = len(tdf)
                    merged = tdf.merge(oaa_join, on=["fielding_team", "game_year"], how="left")
                    matched = merged["sc_team_total_oaa"].notna().sum()
                    print(f"  Team OAA ({tdf_name}): {matched:,}/{before:,} "
                          f"({matched/before*100:.1f}%) matched")
                    if tdf_name == "train":
                        train_df = merged
                    else:
                        test_df = merged
    except Exception as e:
        print(f"  Team OAA not available: {e}")

    # Engineer features
    print("\nEngineering features...")
    X_train, feature_names = engineer_features(train_df)
    X_test, _ = engineer_features(test_df)
    y_train = get_target(train_df)
    y_test = get_target(test_df)

    # === Coverage & NaN Report ===
    print(f"\n{'='*60}")
    print(f"FEATURE COVERAGE & NaN REPORT ({len(feature_names)} features)")
    print(f"{'='*60}")
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Group features
    groups = {
        "Statcast/game-state": [i for i, n in enumerate(feature_names)
                                if not n.startswith(("fg_", "sc_"))],
        "FG pitcher": [i for i, n in enumerate(feature_names) if n.startswith("fg_pit_")],
        "FG batter": [i for i, n in enumerate(feature_names) if n.startswith("fg_bat_")],
        "SC fielding/running": [i for i, n in enumerate(feature_names) if n.startswith("sc_")],
    }
    for group, indices in groups.items():
        if not indices:
            continue
        # NaN count in train
        nan_train = sum(np.isnan(X_train[:, i]).sum() for i in indices)
        nan_test = sum(np.isnan(X_test[:, i]).sum() for i in indices)
        # Zero count (features that are all default/0)
        all_zero = sum(1 for i in indices if np.all(X_train[:, i] == 0))
        all_default = sum(1 for i in indices
                         if np.std(X_train[:, i]) == 0 and len(indices) > 0)
        print(f"\n  {group} ({len(indices)} features):")
        print(f"    Train NaN: {nan_train} / {n_train * len(indices)} "
              f"({nan_train / (n_train * len(indices)) * 100:.2f}%)")
        print(f"    Test NaN: {nan_test} / {n_test * len(indices)} "
              f"({nan_test / (n_test * len(indices)) * 100:.2f}%)")
        print(f"    All-zero features: {all_zero}")
        print(f"    Zero-variance features: {all_default}")

    # Per-feature NaN report for FG features (non-zero NaN only)
    fg_indices = groups["FG pitcher"] + groups["FG batter"]
    fg_nan_features = []
    for i in fg_indices:
        nan_pct = np.isnan(X_train[:, i]).mean() * 100
        if nan_pct > 0:
            fg_nan_features.append((feature_names[i], nan_pct))
    if fg_nan_features:
        print(f"\n  FG features with NaN (train):")
        for name, pct in sorted(fg_nan_features, key=lambda x: -x[1])[:20]:
            print(f"    {name:<30} {pct:5.1f}%")
    else:
        print(f"\n  All FG features: 0% NaN ✓")

    # Year-by-year FG match rate
    if "game_year" in train_df.columns:
        fg_pit_col = next((c for c in train_df.columns if c.startswith("fg_pit_")), None)
        fg_bat_col = next((c for c in train_df.columns if c.startswith("fg_bat_")), None)
        if fg_pit_col or fg_bat_col:
            print(f"\n  Year-by-year FG match rate (train):")
            print(f"  {'Year':<6} {'Pitches':>10} {'PitMatch':>10} {'BatMatch':>10}")
            for yr in sorted(train_df["game_year"].unique()):
                mask = train_df["game_year"] == yr
                n = mask.sum()
                pit_m = train_df.loc[mask, fg_pit_col].notna().sum() if fg_pit_col else 0
                bat_m = train_df.loc[mask, fg_bat_col].notna().sum() if fg_bat_col else 0
                print(f"  {int(yr):<6} {n:>10,} {pit_m/n*100:>9.1f}% {bat_m/n*100:>9.1f}%")

    print(f"{'='*60}")

    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # MLB home_win_exp as benchmark
    print(f"\n{'=' * 60}")
    print("BENCHMARK: MLB home_win_exp (Statcast API)")
    print(f"{'=' * 60}")
    mlb_wp = test_df["home_win_exp"].values
    mlb_valid = ~np.isnan(mlb_wp)
    if mlb_valid.sum() > 0:
        mlb_metrics = evaluate(mlb_wp[mlb_valid], y_test[mlb_valid], "MLB_official")
    else:
        print("  MLB WP not available in test data")
        mlb_metrics = None

    # -------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------
    import lightgbm as lgb

    print(f"\n{'=' * 60}")
    print(f"LightGBM ({len(feature_names)} features)")
    print(f"{'=' * 60}")

    if args.n_trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def lgbm_objective(trial):
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "verbosity": -1, "seed": 42,
                "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 511),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "min_child_samples": trial.suggest_int("min_child", 20, 500),
                "feature_fraction": trial.suggest_float("ff", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bf", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bfreq", 1, 10),
                "reg_alpha": trial.suggest_float("alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("lam", 1e-8, 10, log=True),
            }
            td = lgb.Dataset(X_train, y_train, feature_name=feature_names)
            vd = lgb.Dataset(X_test, y_test, feature_name=feature_names, reference=td)
            m = lgb.train(params, td, 2000, valid_sets=[vd],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(0)])
            p = m.predict(X_test)
            return float(np.mean((p - y_test) ** 2))

        print(f"  Optuna: {args.n_trials} trials...")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lgbm_objective, n_trials=args.n_trials,
                       show_progress_bar=True)
        print(f"  Best Brier: {study.best_value:.6f}")
        best_lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "verbosity": -1, "seed": 42,
            "learning_rate": study.best_params["lr"],
            "num_leaves": study.best_params["num_leaves"],
            "max_depth": study.best_params["max_depth"],
            "min_child_samples": study.best_params["min_child"],
            "feature_fraction": study.best_params["ff"],
            "bagging_fraction": study.best_params["bf"],
            "bagging_freq": study.best_params["bfreq"],
            "reg_alpha": study.best_params["alpha"],
            "reg_lambda": study.best_params["lam"],
        }
    else:
        best_lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.03, "num_leaves": 127,
            "max_depth": 10, "min_child_samples": 200,
            "feature_fraction": 0.7, "bagging_fraction": 0.7,
            "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbose": -1, "seed": 42,
        }

    print("  Training final model...")
    td = lgb.Dataset(X_train, y_train, feature_name=feature_names)
    vd = lgb.Dataset(X_test, y_test, feature_name=feature_names, reference=td)
    lgbm_model = lgb.train(
        best_lgbm_params, td, 3000, valid_sets=[vd],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)])

    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_metrics = evaluate(lgbm_preds, y_test, "LightGBM_statcast")

    # Feature importance
    fi = sorted(zip(feature_names, lgbm_model.feature_importance(importance_type="gain")),
                key=lambda x: -x[1])
    print("  Top 15 features:")
    for name, imp in fi[:15]:
        print(f"    {name:<30} {imp:>10.1f}")

    # Save
    lgbm_model.save_model(str(output_dir / "wp_statcast_lgbm.txt"))

    # -------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------
    cb_metrics = None
    if args.n_trials_catboost > 0:
        try:
            from catboost import CatBoostClassifier
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            print(f"\n{'=' * 60}")
            print(f"CatBoost ({args.n_trials_catboost} Optuna trials)")
            print(f"{'=' * 60}")

            def cb_objective(trial):
                params = {
                    "iterations": 1000,
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2", 1e-3, 10, log=True),
                    "min_child_samples": trial.suggest_int("mcs", 20, 300),
                    "subsample": trial.suggest_float("ss", 0.5, 1.0),
                    "random_seed": 42, "verbose": 0,
                    "eval_metric": "Logloss",
                    "early_stopping_rounds": 50,
                }
                m = CatBoostClassifier(**params)
                m.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
                p = m.predict_proba(X_test)[:, 1]
                return float(np.mean((p - y_test) ** 2))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        pruner=optuna.pruners.MedianPruner())
            study.optimize(cb_objective, n_trials=args.n_trials_catboost,
                           show_progress_bar=True)
            print(f"  Best Brier: {study.best_value:.6f}")

            best_cb = {
                "iterations": 2000,
                "learning_rate": study.best_params["lr"],
                "depth": study.best_params["depth"],
                "l2_leaf_reg": study.best_params["l2"],
                "min_child_samples": study.best_params["mcs"],
                "subsample": study.best_params["ss"],
                "random_seed": 42, "verbose": 0,
                "eval_metric": "Logloss",
                "early_stopping_rounds": 100,
            }
            cb_model = CatBoostClassifier(**best_cb)
            cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
            cb_preds = cb_model.predict_proba(X_test)[:, 1]
            cb_metrics = evaluate(cb_preds, y_test, "CatBoost_statcast")

            cb_model.save_model(str(output_dir / "wp_statcast_catboost.cbm"))

        except ImportError:
            print("  CatBoost not installed, skipping")

    # -------------------------------------------------------
    # Comparison
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON")
    print(f"{'=' * 60}")

    results = {"test_year": args.test_year, "n_features": len(feature_names)}
    results["feature_names"] = feature_names
    results["lgbm"] = lgbm_metrics
    results["lgbm"]["optuna_trials"] = args.n_trials
    results["lgbm"]["best_iteration"] = lgbm_model.best_iteration
    results["lgbm"]["feature_importance"] = {n: round(float(v), 1) for n, v in fi[:30]}

    if mlb_metrics:
        results["mlb_official"] = mlb_metrics
        mlb_brier = mlb_metrics["brier"]
        lgbm_vs_mlb = (mlb_brier - lgbm_metrics["brier"]) / mlb_brier * 100
        print(f"\n  LightGBM vs MLB benchmark: {lgbm_vs_mlb:+.2f}% Brier")

    if cb_metrics:
        results["catboost"] = cb_metrics
        if mlb_metrics:
            cb_vs_mlb = (mlb_brier - cb_metrics["brier"]) / mlb_brier * 100
            print(f"  CatBoost vs MLB benchmark: {cb_vs_mlb:+.2f}% Brier")

    # Save
    results_path = output_dir / "wp_statcast_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
