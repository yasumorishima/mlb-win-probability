"""
MLB Win Probability Engine — Statcast LightGBM Model.

Uses pitch-level and batted-ball features from MLB Stats API live feed
to predict win probability with the trained LightGBM model.

Works in two modes:
  1. Replay/post-game: full Statcast features from completed plays
  2. Live: real-time features from MLB Stats API pitchData/hitData

Same feature engineering as train_wp_statcast.py to ensure consistency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

# Feature names must match training order exactly
FEATURE_NAMES = [
    # Game state (26)
    "inning", "is_bottom", "outs", "balls", "strikes",
    "score_diff", "r1", "r2", "r3", "total_runners",
    "scoring_position", "abs_lead", "game_phase", "walk_off_eligible",
    "tied", "close_game", "count_state", "full_count", "ahead_in_count",
    "inn_x_bottom", "inn_x_outs", "inn_x_score", "outs_x_score",
    "phase_x_lead", "runners_x_outs", "scoring_x_outs",
    # Pitch characteristics (11)
    "release_speed", "effective_speed", "pfx_x", "pfx_z",
    "total_movement", "release_spin", "plate_x", "plate_z",
    "in_zone", "release_ext", "arm_angle",
    # Batted ball (8)
    "launch_speed", "launch_angle", "hit_distance",
    "xwoba", "xba", "xslg", "woba_value", "barrel",
    # Bat tracking (3)
    "bat_speed", "swing_length", "attack_angle_bat",
    # Additional Statcast stats (7)
    "babip_value", "iso_value", "delta_run_exp", "delta_pitcher_run_exp",
    "age_bat", "age_pit", "hit_location",
    # Strike zone / spin / release / trajectory (10)
    "sz_top", "sz_bot", "sz_height", "plate_z_norm",
    "spin_axis", "release_pos_x", "release_pos_z",
    "vx0", "vz0", "ax", "az",
    # FanGraphs pitcher season stats (46)
    "fg_pit_era", "fg_pit_fip", "fg_pit_xfip", "fg_pit_siera",
    "fg_pit_era_minus", "fg_pit_fip_minus", "fg_pit_xfip_minus",
    "fg_pit_k_pct", "fg_pit_bb_pct", "fg_pit_k_bb_pct",
    "fg_pit_k9", "fg_pit_bb9", "fg_pit_k_bb",
    "fg_pit_hr9", "fg_pit_hr_fb", "fg_pit_whip", "fg_pit_babip", "fg_pit_lob_pct",
    "fg_pit_swstr", "fg_pit_csw",
    "fg_pit_o_swing", "fg_pit_z_swing", "fg_pit_o_contact", "fg_pit_z_contact",
    "fg_pit_zone", "fg_pit_fstrike",
    "fg_pit_stuff_plus", "fg_pit_location_plus", "fg_pit_pitching_plus",
    "fg_pit_gb_pct", "fg_pit_fb_pct", "fg_pit_ld_pct", "fg_pit_iffb_pct",
    "fg_pit_pull_pct", "fg_pit_soft_pct", "fg_pit_hard_pct",
    "fg_pit_wfb_c", "fg_pit_wsl_c", "fg_pit_wch_c",
    "fg_pit_gs", "fg_pit_start_ip", "fg_pit_relief_ip",
    "fg_pit_war", "fg_pit_clutch", "fg_pit_wpa", "fg_pit_gmli",
    # FanGraphs batter season stats (35)
    "fg_bat_woba", "fg_bat_xwoba", "fg_bat_wrc_plus", "fg_bat_ops",
    "fg_bat_iso", "fg_bat_babip",
    "fg_bat_k_pct", "fg_bat_bb_pct",
    "fg_bat_o_swing", "fg_bat_z_swing", "fg_bat_o_contact", "fg_bat_z_contact",
    "fg_bat_zone", "fg_bat_swstr", "fg_bat_contact",
    "fg_bat_gb_pct", "fg_bat_fb_pct", "fg_bat_ld_pct", "fg_bat_iffb_pct",
    "fg_bat_hr_fb", "fg_bat_pull_pct",
    "fg_bat_soft_pct", "fg_bat_hard_pct", "fg_bat_hardhit",
    "fg_bat_wfb_c", "fg_bat_wsl_c", "fg_bat_wch_c",
    "fg_bat_spd", "fg_bat_bsr",
    "fg_bat_war", "fg_bat_off", "fg_bat_def",
    "fg_bat_clutch", "fg_bat_wpa", "fg_bat_wraa",
    # Lineup/fatigue + nonlinear + park + clipped (8)
    "n_thruorder", "n_priorpa",
    "score_diff_sq", "inning_sq", "speed_sq", "launch_speed_sq",
    "park_factor", "park_hr_factor",
    "score_capped", "inning_capped",
]


@dataclass
class StatcastWPResult:
    """WP prediction from Statcast LightGBM model."""
    wp: float
    features_used: int      # how many non-zero Statcast features
    has_pitch_data: bool
    has_hit_data: bool
    # Conformal prediction intervals (if quantiles loaded)
    wp_lower_90: float | None = None
    wp_upper_90: float | None = None
    wp_lower_95: float | None = None
    wp_upper_95: float | None = None


class WPEngineStatcast:
    """Statcast LightGBM WP engine."""

    def __init__(self):
        self._model = None
        self._loaded = False
        self._conformal: dict = {}  # quantile label → q value
        self._load()

    def _load(self):
        model_path = DATA_DIR / "wp_statcast_lgbm.txt"
        if not model_path.exists():
            return

        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(model_path))
            # Verify feature count matches
            if self._model.num_feature() == len(FEATURE_NAMES):
                self._loaded = True
            else:
                print(f"WARNING: model has {self._model.num_feature()} features, "
                      f"expected {len(FEATURE_NAMES)}")
        except ImportError:
            pass
        except Exception as e:
            print(f"WARNING: Failed to load Statcast model: {e}")

        # Load conformal prediction quantiles
        conformal_path = DATA_DIR / "conformal_quantiles.json"
        if conformal_path.exists():
            try:
                cq = json.loads(conformal_path.read_text())
                for label, info in cq.get("quantiles", {}).items():
                    self._conformal[label] = info["q"]
            except Exception:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, game_state: dict,
                pitch_data: dict | None = None,
                hit_data: dict | None = None) -> StatcastWPResult | None:
        """
        Predict WP from game state + optional pitch/hit data.

        Args:
            game_state: dict with keys:
                inning, top_bottom, outs, runners (tuple of 3 ints),
                score_diff, balls, strikes
            pitch_data: dict from MLB Stats API pitchData (optional)
                startSpeed, endSpeed, coordinates.pfxX/pfxZ/pX/pZ,
                spinRate, zone, extension
            hit_data: dict from MLB Stats API hitData (optional)
                launchSpeed, launchAngle, totalDistance
        """
        if not self._loaded:
            return None

        features = self._build_features(game_state, pitch_data, hit_data)
        X = np.array([features], dtype=np.float32)
        wp = float(self._model.predict(X)[0])
        wp = max(0.001, min(0.999, wp))

        # Prediction intervals (conformal prediction)
        lo90, hi90, lo95, hi95 = None, None, None, None

        if self._conformal:
            q90 = self._conformal.get("90%")
            q95 = self._conformal.get("95%")
            if q90:
                lo90, hi90 = max(0.0, wp - q90), min(1.0, wp + q90)
            if q95:
                lo95, hi95 = max(0.0, wp - q95), min(1.0, wp + q95)

        return StatcastWPResult(
            wp=wp,
            features_used=sum(1 for f in features[26:46] if f != 0),
            has_pitch_data=pitch_data is not None,
            has_hit_data=hit_data is not None,
            wp_lower_90=lo90,
            wp_upper_90=hi90,
            wp_lower_95=lo95,
            wp_upper_95=hi95,
        )

    def _build_features(self, gs: dict,
                        pitch: dict | None,
                        hit: dict | None) -> list[float]:
        """Build feature vector matching training order."""
        inning = gs.get("inning", 1)
        is_bottom = 1 if gs.get("top_bottom", "top") == "bottom" else 0
        outs = gs.get("outs", 0)
        balls = gs.get("balls", 0)
        strikes = gs.get("strikes", 0)
        score_diff = gs.get("score_diff", 0)

        runners = gs.get("runners", (0, 0, 0))
        r1, r2, r3 = runners[0], runners[1], runners[2]
        total_runners = r1 + r2 + r3
        scoring_position = r2 + r3

        abs_lead = abs(score_diff)
        game_phase = 0 if inning <= 3 else (1 if inning <= 6 else 2)
        walk_off = 1 if (is_bottom and inning >= 9 and score_diff <= 0) else 0
        tied = 1 if score_diff == 0 else 0
        close_game = 1 if abs_lead <= 2 else 0

        count_state = balls * 3 + strikes
        full_count = 1 if (balls == 3 and strikes == 2) else 0
        ahead = 1 if strikes > balls else 0

        # Interactions
        inn_x_bottom = inning * is_bottom
        inn_x_outs = inning * outs
        inn_x_score = inning * score_diff
        outs_x_score = outs * score_diff
        phase_x_lead = game_phase * abs_lead
        runners_x_outs = total_runners * outs
        scoring_x_outs = scoring_position * outs

        # Pitch characteristics (from live feed pitchData)
        release_speed = 0.0
        effective_speed = 0.0
        pfx_x = 0.0
        pfx_z = 0.0
        total_movement = 0.0
        release_spin = 0.0
        plate_x = 0.0
        plate_z = 0.0
        in_zone = 0
        release_ext = 0.0
        arm_angle = 0.0

        if pitch:
            release_speed = _safe(pitch, "startSpeed", 93.0)
            effective_speed = _safe(pitch, "endSpeed", release_speed * 0.92)
            coords = pitch.get("coordinates", {})
            pfx_x = _safe(coords, "pfxX", 0.0)
            pfx_z = _safe(coords, "pfxZ", 0.0)
            total_movement = math.sqrt(pfx_x**2 + pfx_z**2)
            release_spin = _safe(pitch, "spinRate", 2200.0)
            plate_x = _safe(coords, "pX", 0.0)
            plate_z = _safe(coords, "pZ", 2.5)
            zone = _safe(pitch, "zone", 0)
            in_zone = 1 if 1 <= zone <= 9 else 0
            release_ext = _safe(pitch, "extension", 6.0)
            arm_angle = 0.0  # not in live feed

        # Batted ball quality (from live feed hitData)
        launch_speed = 0.0
        launch_angle = 0.0
        hit_distance = 0.0
        xwoba = 0.0
        xba = 0.0
        xslg = 0.0
        woba_value = 0.0
        barrel = 0
        bat_speed = 0.0
        swing_length = 0.0
        attack_angle = 0.0

        if hit:
            launch_speed = _safe(hit, "launchSpeed", 0.0)
            launch_angle = _safe(hit, "launchAngle", 0.0)
            hit_distance = _safe(hit, "totalDistance", 0.0)
            barrel = 1 if (launch_speed >= 98
                           and 26 <= launch_angle <= 30) else 0

        # Additional batting stats (from BQ, 0 in live)
        babip_value = 0.0
        iso_value = 0.0
        delta_run_exp = 0.0
        delta_pitcher_run_exp = 0.0

        # Player age (not in live feed)
        age_bat = gs.get("age_bat", 27.0)
        age_pit = gs.get("age_pit", 27.0)

        # Hit location (not in live feed)
        hit_location = 0.0

        # Strike zone geometry
        sz_top = 3.4
        sz_bot = 1.6
        if pitch:
            sz = pitch.get("strikeZone", {})
            sz_top = _safe(sz, "top", 3.4)
            sz_bot = _safe(sz, "bottom", 1.6)
        sz_height = sz_top - sz_bot
        plate_z_norm = ((plate_z - sz_bot) / sz_height) if sz_height > 0 else 0.5

        # Spin axis
        spin_axis = 0.0
        if pitch:
            spin_axis = _safe(pitch, "spinAxis", 0.0)

        # Release point
        release_pos_x = 0.0
        release_pos_z = 0.0
        if pitch:
            coords = pitch.get("coordinates", {})
            release_pos_x = _safe(coords, "releaseX", 0.0)
            release_pos_z = _safe(coords, "releaseZ", 5.5)

        # Pitch trajectory
        vx0 = 0.0
        vz0 = 0.0
        ax_val = 0.0
        az_val = 0.0
        if pitch:
            vx0 = _safe(pitch, "vX0", 0.0)
            vz0 = _safe(pitch, "vZ0", 0.0)
            ax_val = _safe(pitch, "aX", 0.0)
            az_val = _safe(pitch, "aZ", 0.0)

        # Lineup/fatigue (not available from live feed, use defaults)
        n_thruorder = gs.get("n_thruorder", 1)
        n_priorpa = gs.get("n_priorpa", 0)

        # Nonlinear
        score_diff_sq = score_diff ** 2
        inning_sq = inning ** 2
        speed_sq = release_speed ** 2
        launch_speed_sq = launch_speed ** 2

        # Park factors
        park_factor = gs.get("park_factor", 1.0)
        park_hr_factor = gs.get("park_hr_factor", 1.0)

        # Clipped
        score_capped = max(-10, min(10, score_diff))
        inning_capped = max(1, min(12, inning))

        # FanGraphs pitcher season stats (from gs dict, defaults = league avg)
        fg_pit = gs.get("fg_pit", {})
        fg_pit_era = fg_pit.get("era", 4.0)
        fg_pit_fip = fg_pit.get("fip", 4.0)
        fg_pit_xfip = fg_pit.get("xfip", 4.0)
        fg_pit_siera = fg_pit.get("siera", 4.0)
        fg_pit_era_minus = fg_pit.get("era_minus", 100)
        fg_pit_fip_minus = fg_pit.get("fip_minus", 100)
        fg_pit_xfip_minus = fg_pit.get("xfip_minus", 100)
        fg_pit_k_pct = fg_pit.get("k_pct", 0.22)
        fg_pit_bb_pct = fg_pit.get("bb_pct", 0.08)
        fg_pit_k_bb_pct = fg_pit.get("k_bb_pct", 0.14)
        fg_pit_k9 = fg_pit.get("k9", 8.5)
        fg_pit_bb9 = fg_pit.get("bb9", 3.2)
        fg_pit_k_bb = fg_pit.get("k_bb", 3.0)
        fg_pit_hr9 = fg_pit.get("hr9", 1.1)
        fg_pit_hr_fb = fg_pit.get("hr_fb", 0.11)
        fg_pit_whip = fg_pit.get("whip", 1.3)
        fg_pit_babip = fg_pit.get("babip", 0.29)
        fg_pit_lob_pct = fg_pit.get("lob_pct", 0.72)
        fg_pit_swstr = fg_pit.get("swstr", 0.11)
        fg_pit_csw = fg_pit.get("csw", 0.28)
        fg_pit_o_swing = fg_pit.get("o_swing", 0.32)
        fg_pit_z_swing = fg_pit.get("z_swing", 0.69)
        fg_pit_o_contact = fg_pit.get("o_contact", 0.62)
        fg_pit_z_contact = fg_pit.get("z_contact", 0.86)
        fg_pit_zone = fg_pit.get("zone", 0.43)
        fg_pit_fstrike = fg_pit.get("fstrike", 0.60)
        fg_pit_stuff_plus = fg_pit.get("stuff_plus", 100)
        fg_pit_location_plus = fg_pit.get("location_plus", 100)
        fg_pit_pitching_plus = fg_pit.get("pitching_plus", 100)
        fg_pit_gb_pct = fg_pit.get("gb_pct", 0.42)
        fg_pit_fb_pct = fg_pit.get("fb_pct", 0.38)
        fg_pit_ld_pct = fg_pit.get("ld_pct", 0.19)
        fg_pit_iffb_pct = fg_pit.get("iffb_pct", 0.10)
        fg_pit_pull_pct = fg_pit.get("pull_pct", 0.40)
        fg_pit_soft_pct = fg_pit.get("soft_pct", 0.16)
        fg_pit_hard_pct = fg_pit.get("hard_pct", 0.31)
        fg_pit_wfb_c = fg_pit.get("wfb_c", 0.0)
        fg_pit_wsl_c = fg_pit.get("wsl_c", 0.0)
        fg_pit_wch_c = fg_pit.get("wch_c", 0.0)
        fg_pit_gs = fg_pit.get("gs", 0.0)
        fg_pit_start_ip = fg_pit.get("start_ip", 0.0)
        fg_pit_relief_ip = fg_pit.get("relief_ip", 0.0)
        fg_pit_war = fg_pit.get("war", 0.0)
        fg_pit_clutch = fg_pit.get("clutch", 0.0)
        fg_pit_wpa = fg_pit.get("wpa", 0.0)
        fg_pit_gmli = fg_pit.get("gmli", 1.0)

        # FanGraphs batter season stats
        fg_bat = gs.get("fg_bat", {})
        fg_bat_woba = fg_bat.get("woba", 0.30)
        fg_bat_xwoba = fg_bat.get("xwoba", 0.30)
        fg_bat_wrc_plus = fg_bat.get("wrc_plus", 100)
        fg_bat_ops = fg_bat.get("ops", 0.70)
        fg_bat_iso = fg_bat.get("iso", 0.14)
        fg_bat_babip = fg_bat.get("babip", 0.29)
        fg_bat_k_pct = fg_bat.get("k_pct", 0.22)
        fg_bat_bb_pct = fg_bat.get("bb_pct", 0.08)
        fg_bat_o_swing = fg_bat.get("o_swing", 0.32)
        fg_bat_z_swing = fg_bat.get("z_swing", 0.69)
        fg_bat_o_contact = fg_bat.get("o_contact", 0.61)
        fg_bat_z_contact = fg_bat.get("z_contact", 0.86)
        fg_bat_zone = fg_bat.get("zone", 0.43)
        fg_bat_swstr = fg_bat.get("swstr", 0.11)
        fg_bat_contact = fg_bat.get("contact", 0.76)
        fg_bat_gb_pct = fg_bat.get("gb_pct", 0.43)
        fg_bat_fb_pct = fg_bat.get("fb_pct", 0.38)
        fg_bat_ld_pct = fg_bat.get("ld_pct", 0.20)
        fg_bat_iffb_pct = fg_bat.get("iffb_pct", 0.10)
        fg_bat_hr_fb = fg_bat.get("hr_fb", 0.11)
        fg_bat_pull_pct = fg_bat.get("pull_pct", 0.40)
        fg_bat_soft_pct = fg_bat.get("soft_pct", 0.16)
        fg_bat_hard_pct = fg_bat.get("hard_pct", 0.38)
        fg_bat_hardhit = fg_bat.get("hardhit", 0.38)
        fg_bat_wfb_c = fg_bat.get("wfb_c", 0.0)
        fg_bat_wsl_c = fg_bat.get("wsl_c", 0.0)
        fg_bat_wch_c = fg_bat.get("wch_c", 0.0)
        fg_bat_spd = fg_bat.get("spd", 4.0)
        fg_bat_bsr = fg_bat.get("bsr", 0.0)
        fg_bat_war = fg_bat.get("war", 0.0)
        fg_bat_off = fg_bat.get("off", 0.0)
        fg_bat_def = fg_bat.get("def_", 0.0)
        fg_bat_clutch = fg_bat.get("clutch", 0.0)
        fg_bat_wpa = fg_bat.get("wpa", 0.0)
        fg_bat_wraa = fg_bat.get("wraa", 0.0)

        # Order MUST match engineer_features() in train_wp_statcast.py
        return [
            inning, is_bottom, outs, balls, strikes,
            score_diff, r1, r2, r3, total_runners,
            scoring_position, abs_lead, game_phase, walk_off,
            tied, close_game, count_state, full_count, ahead,
            inn_x_bottom, inn_x_outs, inn_x_score, outs_x_score,
            phase_x_lead, runners_x_outs, scoring_x_outs,
            release_speed, effective_speed, pfx_x, pfx_z,
            total_movement, release_spin, plate_x, plate_z,
            in_zone, release_ext, arm_angle,
            launch_speed, launch_angle, hit_distance,
            xwoba, xba, xslg, woba_value,
            barrel, bat_speed, swing_length, attack_angle,
            babip_value, iso_value, delta_run_exp, delta_pitcher_run_exp,
            age_bat, age_pit,
            hit_location,
            sz_top, sz_bot, sz_height, plate_z_norm,
            spin_axis,
            release_pos_x, release_pos_z,
            vx0, vz0, ax_val, az_val,
            # FG pitcher (46)
            fg_pit_era, fg_pit_fip, fg_pit_xfip, fg_pit_siera,
            fg_pit_era_minus, fg_pit_fip_minus, fg_pit_xfip_minus,
            fg_pit_k_pct, fg_pit_bb_pct, fg_pit_k_bb_pct,
            fg_pit_k9, fg_pit_bb9, fg_pit_k_bb,
            fg_pit_hr9, fg_pit_hr_fb, fg_pit_whip, fg_pit_babip, fg_pit_lob_pct,
            fg_pit_swstr, fg_pit_csw,
            fg_pit_o_swing, fg_pit_z_swing, fg_pit_o_contact, fg_pit_z_contact,
            fg_pit_zone, fg_pit_fstrike,
            fg_pit_stuff_plus, fg_pit_location_plus, fg_pit_pitching_plus,
            fg_pit_gb_pct, fg_pit_fb_pct, fg_pit_ld_pct, fg_pit_iffb_pct,
            fg_pit_pull_pct, fg_pit_soft_pct, fg_pit_hard_pct,
            fg_pit_wfb_c, fg_pit_wsl_c, fg_pit_wch_c,
            fg_pit_gs, fg_pit_start_ip, fg_pit_relief_ip,
            fg_pit_war, fg_pit_clutch, fg_pit_wpa, fg_pit_gmli,
            # FG batter (35)
            fg_bat_woba, fg_bat_xwoba, fg_bat_wrc_plus, fg_bat_ops,
            fg_bat_iso, fg_bat_babip,
            fg_bat_k_pct, fg_bat_bb_pct,
            fg_bat_o_swing, fg_bat_z_swing, fg_bat_o_contact, fg_bat_z_contact,
            fg_bat_zone, fg_bat_swstr, fg_bat_contact,
            fg_bat_gb_pct, fg_bat_fb_pct, fg_bat_ld_pct, fg_bat_iffb_pct,
            fg_bat_hr_fb, fg_bat_pull_pct,
            fg_bat_soft_pct, fg_bat_hard_pct, fg_bat_hardhit,
            fg_bat_wfb_c, fg_bat_wsl_c, fg_bat_wch_c,
            fg_bat_spd, fg_bat_bsr,
            fg_bat_war, fg_bat_off, fg_bat_def,
            fg_bat_clutch, fg_bat_wpa, fg_bat_wraa,
            # Lineup/fatigue + nonlinear + park + clipped
            n_thruorder, n_priorpa,
            score_diff_sq, inning_sq, speed_sq, launch_speed_sq,
            park_factor, park_hr_factor,
            score_capped, inning_capped,
        ]


def _safe(d: dict, key: str, default: float = 0.0) -> float:
    """Safely extract a numeric value from dict."""
    val = d.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
