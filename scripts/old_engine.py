"""
The old hand-tuned win-probability function (normal approximation over
RE24, Optuna-tuned constants), kept only so scripts/fit_engine.py can score
it next to the fitted engine on the held-out season
(reports/engine_check_*.json). The app and the API do not use it.

Copied verbatim from win_probability.py as of commit 0769c19.
"""

import numpy as np
from scipy.stats import norm


RE24_MLB: dict[tuple[int, int, int, int], float] = {
    # --- 0 outs ---
    (0, 0, 0, 0): 0.481,
    (1, 0, 0, 0): 0.859,
    (0, 1, 0, 0): 1.100,
    (1, 1, 0, 0): 1.437,
    (0, 0, 1, 0): 1.350,
    (1, 0, 1, 0): 1.784,
    (0, 1, 1, 0): 1.964,
    (1, 1, 1, 0): 2.292,
    # --- 1 out ---
    (0, 0, 0, 1): 0.254,
    (1, 0, 0, 1): 0.509,
    (0, 1, 0, 1): 0.664,
    (1, 1, 0, 1): 0.884,
    (0, 0, 1, 1): 0.950,
    (1, 0, 1, 1): 1.130,
    (0, 1, 1, 1): 1.376,
    (1, 1, 1, 1): 1.541,
    # --- 2 outs ---
    (0, 0, 0, 2): 0.098,
    (1, 0, 0, 2): 0.224,
    (0, 1, 0, 2): 0.319,
    (1, 1, 0, 2): 0.429,
    (0, 0, 1, 2): 0.353,
    (1, 0, 1, 2): 0.478,
    (0, 1, 1, 2): 0.580,
    (1, 1, 1, 2): 0.752,
}

MLB_RPG = 4.5  # MLB average runs per game (per team)


def get_re24(runners: tuple[int, int, int], outs: int,
             runs_per_game: float = MLB_RPG) -> float:
    """Get Run Expectancy for a given base-out state, scaled by scoring environment."""
    key = (runners[0], runners[1], runners[2], outs)
    base = RE24_MLB.get(key, 0.0)
    return base * (runs_per_game / MLB_RPG)


def _remaining_innings(inning: int, top_bottom: str) -> float:
    """Calculate remaining half-innings for the home team to score."""
    if top_bottom == "top":
        # Home team bats in bottom half; remaining = (9 - inning + 1) bottoms
        return max(9 - inning + 1, 0.5)
    else:
        # Currently bottom; remaining = (9 - inning) full innings + current
        return max(9 - inning + 0.5, 0.5)


def calculate_wp(inning: int, top_bottom: str, outs: int,
                 runners: tuple[int, int, int], score_diff: int,
                 runs_per_game: float = MLB_RPG) -> float:
    """
    Calculate Win Probability for the home team.

    Parameters:
        inning: Current inning (1-9+)
        top_bottom: "top" or "bottom"
        outs: Number of outs (0-2)
        runners: (1B, 2B, 3B) where 1=occupied, 0=empty
        score_diff: Home score minus Away score
        runs_per_game: Scoring environment (4.5 for MLB, 4.0 for NPB)

    Returns:
        Win probability for home team (0.0 to 1.0)
    """
    re = get_re24(runners, outs, runs_per_game)
    remaining = _remaining_innings(inning, top_bottom)
    rpg = runs_per_game

    # Expected additional runs from current state
    if top_bottom == "top":
        # Visiting team is batting - RE benefits them
        away_extra = re
        home_extra = 0.0
    else:
        away_extra = 0.0
        home_extra = re

    # Remaining runs from future innings (normal approximation)
    # Each half-inning: mean = rpg/18, std = sqrt(rpg/18 * variance_factor)
    runs_per_half = rpg / 18.0
    variance_factor = 3.66  # Optuna-optimized (was 1.3)

    # Total remaining half-innings for each team
    if top_bottom == "top":
        away_remaining_halves = max(9 - inning, 0)  # bottom of current excluded (away done after this)
        home_remaining_halves = max(9 - inning + 1, 1)  # home still bats this inning + rest
    else:
        away_remaining_halves = max(9 - inning, 0)
        home_remaining_halves = max(9 - inning, 0)

    home_mean = score_diff + home_extra - away_extra + (home_remaining_halves - away_remaining_halves) * runs_per_half
    home_std = np.sqrt((home_remaining_halves + away_remaining_halves) * runs_per_half * variance_factor + 0.01)

    # Late-inning adjustments
    if inning >= 9 and top_bottom == "bottom":
        # Bottom of 9th or later - home team only needs to take lead (walk-off)
        if score_diff > 0:
            return 0.99  # Already winning, game should be over
        if score_diff == 0:
            # Tie game in bottom 9th+: probability of scoring at least 1 run
            # Use RE24 as expected runs; P(score >= 1) from Poisson-like model
            # With bases loaded 2 out RE=0.752, P(score>=1) is high
            # Empirical: P(scoring >= 1 run) ≈ 1 - e^(-RE * scoring_factor)
            scoring_factor = 0.87  # Optuna-optimized (was 1.8)
            p_score_1 = 1.0 - np.exp(-re * scoring_factor)
            # Home also has advantage of extra innings if they don't score
            p_extras_win = 0.41  # Optuna-optimized (was 0.50)
            wp = p_score_1 + (1 - p_score_1) * p_extras_win
            return min(0.99, max(0.01, wp))
        else:
            # Behind - need to overcome deficit
            needed = abs(score_diff)
            # P(scoring >= needed runs) from current RE + Poisson approximation
            # More aggressive lambda for loaded bases situations
            lam = re * 0.45  # Optuna-optimized behind_lambda_mult (was 1.5)
            # Poisson CDF: P(X >= needed)
            from scipy.stats import poisson
            p_enough = 1.0 - poisson.cdf(needed - 1, lam)
            # Plus chance of extras if they tie
            p_tie = poisson.pmf(needed, lam) * 0.0  # already counted in p_enough
            return min(0.99, max(0.01, p_enough))

    if inning >= 9 and top_bottom == "top":
        if score_diff > 0:
            # Home leading, visitor batting in 9th+
            # P(visitor ties or takes lead) based on current RE
            lam = re * 0.67  # Optuna-optimized top9_lambda_mult (was 1.3)
            from scipy.stats import poisson
            p_tie_or_lead = 1.0 - poisson.cdf(score_diff - 1, lam)
            p_lead = 1.0 - poisson.cdf(score_diff, lam)
            # If they tie, home still has ~55% in walk-off bottom
            p_home_loses = p_lead + (p_tie_or_lead - p_lead) * 0.45
            return min(0.99, max(0.01, 1.0 - p_home_loses))

    # Normal approximation for earlier innings
    wp = norm.cdf(home_mean / max(home_std, 0.1))
    return min(0.99, max(0.01, float(wp)))
