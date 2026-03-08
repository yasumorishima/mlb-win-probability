"""
MLB Win Probability Engine v2 — Empirical WP Table + Markov Chain

Replaces the Normal approximation (v1) with:
  A) Empirical WP table: 10,000+ states from 10 years of MLB play-by-play
  B) Markov chain fallback: half-inning run distributions for unseen states

Same API as v1 for drop-in replacement.
Falls back to v1 if data files are not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from win_probability import (
    # Reuse v1 components that don't depend on WP calculation method
    get_re24,
    get_re24_table,
    li_label,
    calculate_wpa,
    GameState,
    get_tactical_recommendations,
    adjust_wp_for_matchup,
    SCENARIOS,
    analyze_scenario,
    MLB_RPG,
    NPB_RPG,
    OUTCOME_PROBS,
    LEAGUE_AVG_WP_SWING,
    _apply_outcome,
)

DATA_DIR = Path(__file__).parent / "data"
MAX_RUNS = 15
EXTRAS_HOME_WIN = 0.52


class WPEngineV2:
    """Win Probability engine using empirical table + Markov chain."""

    def __init__(self, data_dir: Path | str | None = None):
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.wp_table: dict = {}
        self.run_dists: dict[str, np.ndarray] = {}
        self._loaded = False

        wp_path = data_dir / "empirical_wp_table.json"
        rd_path = data_dir / "halfinn_run_dist.json"

        if wp_path.exists() and rd_path.exists():
            with open(wp_path) as f:
                self.wp_table = json.load(f)
            with open(rd_path) as f:
                rd_raw = json.load(f)
                self.run_dists = {k: np.array(v) for k, v in rd_raw.items()}
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ----------------------------------------------------------
    # Core WP calculation
    # ----------------------------------------------------------

    def calculate_wp(self, inning: int, top_bottom: str, outs: int,
                     runners: tuple[int, int, int], score_diff: int,
                     runs_per_game: float = MLB_RPG) -> float:
        """Calculate Win Probability using empirical table + Markov fallback.

        Parameters match v1's calculate_wp() signature for drop-in replacement.
        """
        if not self._loaded:
            from win_probability import calculate_wp as v1_wp
            return v1_wp(inning, top_bottom, outs, runners,
                         score_diff, runs_per_game)

        r1, r2, r3 = runners
        key = self._wp_state_key(inning, top_bottom, outs,
                                  r1, r2, r3, score_diff)

        entry = self.wp_table.get(key)

        if entry and entry["n"] >= 10:
            wp = entry["wp"]

            # Adjust for non-MLB scoring environments (e.g. NPB 4.0 R/G)
            if abs(runs_per_game - MLB_RPG) > 0.1:
                wp = self._adjust_for_environment(wp, runs_per_game)

            return max(0.01, min(0.99, float(wp)))

        # Fallback: Markov chain computation
        return self._markov_wp(inning, top_bottom, outs,
                               r1, r2, r3, score_diff)

    # ----------------------------------------------------------
    # Empirical table helpers
    # ----------------------------------------------------------

    @staticmethod
    def _wp_state_key(inning: int, half: str,
                      outs: int, r1: int, r2: int, r3: int,
                      score_diff: int) -> str:
        inn_cap = min(inning, 10)
        diff_cap = max(-10, min(10, score_diff))
        return f"{inn_cap}_{half}_{outs}_{r1}{r2}{r3}_{diff_cap}"

    @staticmethod
    def _adjust_for_environment(wp: float, rpg: float) -> float:
        """Scale WP for different scoring environments via logit adjustment."""
        p = max(0.01, min(0.99, wp))
        logit = np.log(p / (1 - p))
        # Mild scaling: lower scoring compresses toward 0.5
        ratio = rpg / MLB_RPG
        logit *= ratio ** 0.3
        return float(1.0 / (1.0 + np.exp(-logit)))

    # ----------------------------------------------------------
    # Markov chain WP (fallback)
    # ----------------------------------------------------------

    def _markov_wp(self, inning: int, top_bottom: str,
                   outs: int, r1: int, r2: int, r3: int,
                   score_diff: int) -> float:
        """Compute WP by convolving remaining half-inning run distributions."""
        current_key = f"{outs}_{r1}{r2}{r3}"
        fresh_key = "0_000"

        if current_key not in self.run_dists:
            current_key = fresh_key

        max_len = MAX_RUNS + 1
        current_dist = self.run_dists.get(current_key,
                                           np.zeros(max_len))[:max_len]
        fresh_dist = self.run_dists.get(fresh_key,
                                         np.zeros(max_len))[:max_len]

        if top_bottom == "top":
            away_future = max(9 - inning, 0)
            home_future = max(9 - inning + 1, 1)
            away_dists = [current_dist] + [fresh_dist] * away_future
            home_dists = [fresh_dist] * home_future
        else:
            home_future = max(9 - inning, 0)
            away_future = max(9 - inning, 0)
            home_dists = [current_dist] + [fresh_dist] * home_future
            away_dists = ([fresh_dist] * away_future
                          if away_future > 0 else [])

        home_total = self._convolve(home_dists, max_len)
        away_total = self._convolve(away_dists, max_len)

        wp = 0.0
        for h in range(max_len):
            if home_total[h] < 1e-10:
                continue
            for a in range(max_len):
                if away_total[a] < 1e-10:
                    continue
                diff = score_diff + h - a
                if diff > 0:
                    wp += home_total[h] * away_total[a]
                elif diff == 0:
                    wp += home_total[h] * away_total[a] * EXTRAS_HOME_WIN

        return max(0.01, min(0.99, wp))

    @staticmethod
    def _convolve(dists: list[np.ndarray],
                  max_len: int) -> np.ndarray:
        if not dists:
            result = np.zeros(max_len)
            result[0] = 1.0
            return result
        result = dists[0].copy()
        for d in dists[1:]:
            result = np.convolve(result, d)[:max_len]
        s = result.sum()
        if s > 0:
            result /= s
        return result


# ============================================================
# Module-level singleton for convenience
# ============================================================

_engine: WPEngineV2 | None = None


def _get_engine() -> WPEngineV2:
    global _engine
    if _engine is None:
        _engine = WPEngineV2()
    return _engine


def calculate_wp_v2(inning: int, top_bottom: str, outs: int,
                    runners: tuple[int, int, int], score_diff: int,
                    runs_per_game: float = MLB_RPG) -> float:
    """Module-level WP calculation using v2 engine."""
    return _get_engine().calculate_wp(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)


def calculate_li_v2(inning: int, top_bottom: str, outs: int,
                    runners: tuple[int, int, int], score_diff: int,
                    runs_per_game: float = MLB_RPG) -> float:
    """Leverage Index using v2 WP engine."""
    engine = _get_engine()
    base_wp = engine.calculate_wp(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)

    total_wp_swing = 0.0
    for outcome, prob in OUTCOME_PROBS.items():
        new_runners, new_outs, runs = _apply_outcome(runners, outs, outcome)

        if new_outs >= 3:
            if top_bottom == "top":
                new_wp = engine.calculate_wp(
                    inning, "bottom", 0, (0, 0, 0),
                    score_diff - runs, runs_per_game)
            else:
                new_wp = engine.calculate_wp(
                    inning + 1, "top", 0, (0, 0, 0),
                    score_diff + runs, runs_per_game)
        else:
            if top_bottom == "top":
                new_wp = engine.calculate_wp(
                    inning, top_bottom, new_outs, new_runners,
                    score_diff - runs, runs_per_game)
            else:
                new_wp = engine.calculate_wp(
                    inning, top_bottom, new_outs, new_runners,
                    score_diff + runs, runs_per_game)

        total_wp_swing += prob * abs(new_wp - base_wp)

    return round(total_wp_swing / LEAGUE_AVG_WP_SWING, 2)


def full_analysis_v2(inning: int, top_bottom: str, outs: int,
                     runners: tuple[int, int, int], score_diff: int,
                     runs_per_game: float = MLB_RPG,
                     batter_ops: float | None = None,
                     pitcher_era: float | None = None) -> dict:
    """Full analysis (WP + LI + tactics) using v2 engine."""
    engine = _get_engine()
    wp = engine.calculate_wp(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)
    li = calculate_li_v2(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)
    tactics = get_tactical_recommendations(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)

    adjusted_wp = adjust_wp_for_matchup(wp, batter_ops, pitcher_era)

    return {
        "game_state": {
            "inning": inning,
            "top_bottom": top_bottom,
            "outs": outs,
            "runners": {"1B": runners[0], "2B": runners[1], "3B": runners[2]},
            "score_diff": score_diff,
            "runs_per_game": runs_per_game,
        },
        "win_probability": round(wp, 4),
        "win_probability_pct": f"{wp * 100:.1f}%",
        "adjusted_wp": (round(adjusted_wp, 4)
                        if (batter_ops or pitcher_era) else None),
        "leverage_index": li,
        "leverage_label": li_label(li),
        "tactics": tactics,
        "engine": "v2_empirical_markov",
    }
