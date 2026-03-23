"""
MLB Win Probability Engine — Bayesian Hierarchical Model.

Uses posterior parameters from NumPyro SVI training to compute WP with:
  - Team-specific strength adjustments
  - Park effects with Bayesian shrinkage
  - Season/era trend corrections
  - Leverage × team interaction
  - 90% credible intervals on every prediction

Same API as v1/v2 for drop-in integration into the ensemble.
Falls back to v1 if posterior files are not available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from win_probability import (
    MLB_RPG,
    NPB_RPG,
    calculate_wp as v1_calculate_wp,
    li_label,
    get_tactical_recommendations,
    adjust_wp_for_matchup,
    OUTCOME_PROBS,
    LEAGUE_AVG_WP_SWING,
    _apply_outcome,
)

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class BayesianWPResult:
    """WP prediction with uncertainty quantification."""
    wp: float              # posterior mean WP
    wp_lower: float        # 5th percentile (90% CI lower)
    wp_upper: float        # 95th percentile (90% CI upper)
    team_effect: float     # home - away team strength contribution
    park_effect: float     # park effect contribution
    season_effect: float   # season/era contribution
    ci_width: float        # upper - lower (uncertainty magnitude)


class WPEngineBayesian:
    """Bayesian Hierarchical Win Probability engine."""

    def __init__(self, data_dir: Path | str | None = None):
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._loaded = False

        # Load posterior summary
        posterior_path = data_dir / "bayesian_posterior.json"
        samples_path = data_dir / "bayesian_posterior_samples.json"

        if not posterior_path.exists():
            return

        with open(posterior_path) as f:
            self._params = json.load(f)

        # Load full posterior samples for CI computation
        self._has_samples = False
        if samples_path.exists():
            with open(samples_path) as f:
                self._samples = json.load(f)
            self._has_samples = True

        # Build team lookup
        self._teams = self._params["teams"]
        self._team_to_idx = {t: i for i, t in enumerate(self._teams)}
        self._n_teams = self._params["n_teams"]

        # Posterior means (for fast point prediction)
        self._team_strength = np.array(self._params["team_strength_mean"])
        self._team_strength_std = np.array(self._params["team_strength_std"])
        self._park_effect = np.array(self._params["park_effect_mean"])
        self._park_effect_std = np.array(self._params["park_effect_std"])
        self._season_effect = np.array(self._params["season_effect_mean"])
        self._season_effect_std = np.array(self._params["season_effect_std"])
        self._kappa = self._params["kappa_mean"]
        self._kappa_std = self._params["kappa_std"]
        self._hfa_correction = self._params["hfa_correction_mean"]
        self._hfa_correction_std = self._params["hfa_correction_std"]
        self._leverage_mean = self._params["leverage_mean"]
        self._leverage_std = self._params["leverage_std"]
        self._train_years = self._params["train_years"]

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def teams(self) -> list[str]:
        return self._teams if self._loaded else []

    # ----------------------------------------------------------
    # Leverage computation (same as training)
    # ----------------------------------------------------------

    @staticmethod
    def _compute_leverage(inning: int, half: str, outs: int,
                          r1: int, r2: int, r3: int,
                          score_diff: int) -> float:
        """Approximate leverage index."""
        inn_factor = min(inning / 9.0, 1.5)
        abs_diff = abs(score_diff)
        close_factor = max(0, 1.0 - abs_diff / 5.0)
        runner_factor = 1.0 + 0.3 * (r1 + r2 + r3)
        out_factor = 1.0 - 0.2 * outs
        li = inn_factor * close_factor * runner_factor * out_factor
        if inning >= 7 and abs_diff <= 2:
            li *= 1.5
        if inning >= 9 and half == "bottom" and abs_diff <= 1:
            li *= 2.0
        return max(0.1, min(li, 5.0))

    # ----------------------------------------------------------
    # Core prediction
    # ----------------------------------------------------------

    def calculate_wp(self, inning: int, top_bottom: str, outs: int,
                     runners: tuple[int, int, int], score_diff: int,
                     runs_per_game: float = MLB_RPG,
                     home_team: str | None = None,
                     away_team: str | None = None,
                     ) -> float:
        """Calculate WP using Bayesian posterior mean.

        If home_team/away_team are provided, applies team-specific adjustments.
        Otherwise falls back to baseline + HFA correction only.
        Compatible with v1/v2 API (extra args are optional).
        """
        if not self._loaded:
            return v1_calculate_wp(inning, top_bottom, outs, runners,
                                   score_diff, runs_per_game)

        # Markov baseline (v1)
        base_wp = v1_calculate_wp(inning, top_bottom, outs, runners,
                                  score_diff, runs_per_game)
        logit_base = np.log(
            np.clip(base_wp, 0.005, 0.995)
            / (1 - np.clip(base_wp, 0.005, 0.995))
        )

        logit_wp = logit_base + self._hfa_correction

        if home_team and away_team:
            h_idx = self._team_to_idx.get(home_team)
            a_idx = self._team_to_idx.get(away_team)

            if h_idx is not None and a_idx is not None:
                team_diff = (self._team_strength[h_idx]
                             - self._team_strength[a_idx])
                logit_wp += team_diff
                logit_wp += self._park_effect[h_idx]

                # Leverage interaction
                r1, r2, r3 = runners
                li = self._compute_leverage(
                    inning, top_bottom, outs, r1, r2, r3, score_diff)
                li_norm = (li - self._leverage_mean) / self._leverage_std
                logit_wp += self._kappa * li_norm * team_diff

        wp = 1.0 / (1.0 + np.exp(-logit_wp))
        return float(np.clip(wp, 0.01, 0.99))

    def calculate_wp_with_ci(self, inning: int, top_bottom: str, outs: int,
                             runners: tuple[int, int, int], score_diff: int,
                             runs_per_game: float = MLB_RPG,
                             home_team: str | None = None,
                             away_team: str | None = None,
                             n_samples: int = 200,
                             ) -> BayesianWPResult:
        """Calculate WP with 90% credible interval from posterior samples.

        This is the key differentiator: uncertainty that widens in
        high-leverage, close-game situations where the outcome is
        genuinely uncertain.
        """
        if not self._loaded or not self._has_samples:
            wp = self.calculate_wp(
                inning, top_bottom, outs, runners, score_diff,
                runs_per_game, home_team, away_team)
            return BayesianWPResult(
                wp=wp, wp_lower=wp, wp_upper=wp,
                team_effect=0, park_effect=0, season_effect=0,
                ci_width=0)

        # Markov baseline
        base_wp = v1_calculate_wp(inning, top_bottom, outs, runners,
                                  score_diff, runs_per_game)
        logit_base = np.log(
            np.clip(base_wp, 0.005, 0.995)
            / (1 - np.clip(base_wp, 0.005, 0.995))
        )

        # Leverage
        r1, r2, r3 = runners
        li = self._compute_leverage(
            inning, top_bottom, outs, r1, r2, r3, score_diff)
        li_norm = (li - self._leverage_mean) / self._leverage_std

        # Sample from posterior
        team_str_samples = np.array(self._samples["team_strength"])
        park_samples = np.array(self._samples["park_effect"])
        season_samples = np.array(self._samples["season_effect"])
        kappa_samples = np.array(self._samples["kappa"])
        hfa_samples = np.array(self._samples["hfa_correction"])

        total_samples = len(kappa_samples)
        sample_idx = np.random.choice(
            total_samples, size=min(n_samples, total_samples), replace=False)

        wp_samples = np.zeros(len(sample_idx))
        team_effects = np.zeros(len(sample_idx))
        park_effects = np.zeros(len(sample_idx))

        h_idx = self._team_to_idx.get(home_team or "") if home_team else None
        a_idx = self._team_to_idx.get(away_team or "") if away_team else None

        for i, si in enumerate(sample_idx):
            logit_wp = logit_base + hfa_samples[si]
            te = 0.0
            pe = 0.0

            if h_idx is not None and a_idx is not None:
                team_diff = (team_str_samples[si][h_idx]
                             - team_str_samples[si][a_idx])
                te = team_diff
                pe = park_samples[si][h_idx]
                logit_wp += team_diff + pe
                logit_wp += kappa_samples[si] * li_norm * team_diff

            wp_samples[i] = 1.0 / (1.0 + np.exp(-logit_wp))
            team_effects[i] = te
            park_effects[i] = pe

        mean_wp = float(np.clip(np.mean(wp_samples), 0.01, 0.99))
        lower = float(np.clip(np.percentile(wp_samples, 5), 0.01, 0.99))
        upper = float(np.clip(np.percentile(wp_samples, 95), 0.01, 0.99))

        return BayesianWPResult(
            wp=mean_wp,
            wp_lower=lower,
            wp_upper=upper,
            team_effect=float(np.mean(team_effects)),
            park_effect=float(np.mean(park_effects)),
            season_effect=float(self._season_effect[-1])
            if len(self._season_effect) > 0 else 0.0,
            ci_width=upper - lower,
        )

    # ----------------------------------------------------------
    # Team rankings
    # ----------------------------------------------------------

    def get_team_rankings(self) -> list[dict]:
        """Return team strength rankings with uncertainty."""
        if not self._loaded:
            return []

        rankings = []
        for i, team in enumerate(self._teams):
            strength = self._team_strength[i]
            # Convert logit-scale to WP impact (vs average opponent)
            wp_vs_avg = float(1 / (1 + np.exp(-strength)) - 0.5)
            rankings.append({
                "team": team,
                "strength": round(float(strength), 4),
                "strength_std": round(float(self._team_strength_std[i]), 4),
                "wp_vs_avg": round(wp_vs_avg, 3),
                "park_effect": round(float(self._park_effect[i]), 4),
                "park_std": round(float(self._park_effect_std[i]), 4),
            })

        rankings.sort(key=lambda x: -x["strength"])
        return rankings


# ============================================================
# Module-level singleton
# ============================================================

_engine: WPEngineBayesian | None = None


def _get_engine() -> WPEngineBayesian:
    global _engine
    if _engine is None:
        _engine = WPEngineBayesian()
    return _engine


def calculate_wp_bayesian(inning: int, top_bottom: str, outs: int,
                          runners: tuple[int, int, int], score_diff: int,
                          runs_per_game: float = MLB_RPG,
                          home_team: str | None = None,
                          away_team: str | None = None) -> float:
    """Module-level WP calculation using Bayesian engine."""
    return _get_engine().calculate_wp(
        inning, top_bottom, outs, runners, score_diff,
        runs_per_game, home_team, away_team)


def calculate_wp_with_ci(inning: int, top_bottom: str, outs: int,
                         runners: tuple[int, int, int], score_diff: int,
                         runs_per_game: float = MLB_RPG,
                         home_team: str | None = None,
                         away_team: str | None = None,
                         ) -> BayesianWPResult:
    """Module-level WP calculation with credible intervals."""
    return _get_engine().calculate_wp_with_ci(
        inning, top_bottom, outs, runners, score_diff,
        runs_per_game, home_team, away_team)


def full_analysis_bayesian(inning: int, top_bottom: str, outs: int,
                           runners: tuple[int, int, int], score_diff: int,
                           runs_per_game: float = MLB_RPG,
                           home_team: str | None = None,
                           away_team: str | None = None,
                           batter_ops: float | None = None,
                           pitcher_era: float | None = None) -> dict:
    """Full analysis with Bayesian uncertainty."""
    engine = _get_engine()
    result = engine.calculate_wp_with_ci(
        inning, top_bottom, outs, runners, score_diff,
        runs_per_game, home_team, away_team)

    li_val = engine._compute_leverage(
        inning, top_bottom, outs,
        runners[0], runners[1], runners[2], score_diff)

    tactics = get_tactical_recommendations(
        inning, top_bottom, outs, runners, score_diff, runs_per_game)

    adjusted_wp = adjust_wp_for_matchup(result.wp, batter_ops, pitcher_era)

    return {
        "game_state": {
            "inning": inning,
            "top_bottom": top_bottom,
            "outs": outs,
            "runners": {"1B": runners[0], "2B": runners[1], "3B": runners[2]},
            "score_diff": score_diff,
            "runs_per_game": runs_per_game,
            "home_team": home_team,
            "away_team": away_team,
        },
        "win_probability": round(result.wp, 4),
        "win_probability_pct": f"{result.wp * 100:.1f}%",
        "credible_interval": {
            "lower": round(result.wp_lower, 4),
            "upper": round(result.wp_upper, 4),
            "width": round(result.ci_width, 4),
            "level": "90%",
        },
        "effects": {
            "team": round(result.team_effect, 4),
            "park": round(result.park_effect, 4),
            "season": round(result.season_effect, 4),
        },
        "adjusted_wp": (round(adjusted_wp, 4)
                        if (batter_ops or pitcher_era) else None),
        "leverage_index": round(li_val, 2),
        "leverage_label": li_label(li_val),
        "tactics": tactics,
        "engine": "bayesian_hierarchical",
    }
