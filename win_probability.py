"""
MLB Win Probability Engine

Markov Chain + RE24 approach for calculating:
- Win Probability (WP)
- Win Probability Added (WPA)
- Leverage Index (LI)
- Tactical recommendations based on RE24 expected value changes

No external data dependencies - pure mathematical model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ============================================================
# RE24 Table (Run Expectancy by 24 base-out states)
# ============================================================
# MLB 2010-2019 average values
# Key: (runner1, runner2, runner3, outs) -> expected runs

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
NPB_RPG = 4.0  # NPB average runs per game (per team)


def get_re24(runners: tuple[int, int, int], outs: int,
             runs_per_game: float = MLB_RPG) -> float:
    """Get Run Expectancy for a given base-out state, scaled by scoring environment."""
    key = (runners[0], runners[1], runners[2], outs)
    base = RE24_MLB.get(key, 0.0)
    return base * (runs_per_game / MLB_RPG)


def get_re24_table(runs_per_game: float = MLB_RPG) -> list[dict]:
    """Return full RE24 table as a list of dicts."""
    runner_labels = {
        (0, 0, 0): "---",
        (1, 0, 0): "1--",
        (0, 1, 0): "-2-",
        (1, 1, 0): "12-",
        (0, 0, 1): "--3",
        (1, 0, 1): "1-3",
        (0, 1, 1): "-23",
        (1, 1, 1): "123",
    }
    table = []
    for (r1, r2, r3, outs), base_re in sorted(RE24_MLB.items()):
        scaled = base_re * (runs_per_game / MLB_RPG)
        table.append({
            "runners": runner_labels[(r1, r2, r3)],
            "runner1": r1,
            "runner2": r2,
            "runner3": r3,
            "outs": outs,
            "expected_runs": round(scaled, 3),
        })
    return table


# ============================================================
# Win Probability Calculation (Markov Chain + Normal Approx)
# ============================================================


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
    variance_factor = 1.3  # empirical variance adjustment

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
            scoring_factor = 1.8  # calibrated to match historical WP tables
            p_score_1 = 1.0 - np.exp(-re * scoring_factor)
            # Home also has advantage of extra innings if they don't score
            p_extras_win = 0.50  # roughly 50/50 in extras
            wp = p_score_1 + (1 - p_score_1) * p_extras_win
            return min(0.99, max(0.01, wp))
        else:
            # Behind - need to overcome deficit
            needed = abs(score_diff)
            # P(scoring >= needed runs) from current RE + Poisson approximation
            # More aggressive lambda for loaded bases situations
            lam = re * 1.5
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
            lam = re * 1.3
            from scipy.stats import poisson
            p_tie_or_lead = 1.0 - poisson.cdf(score_diff - 1, lam)
            p_lead = 1.0 - poisson.cdf(score_diff, lam)
            # If they tie, home still has ~55% in walk-off bottom
            p_home_loses = p_lead + (p_tie_or_lead - p_lead) * 0.45
            return min(0.99, max(0.01, 1.0 - p_home_loses))

    # Normal approximation for earlier innings
    wp = norm.cdf(home_mean / max(home_std, 0.1))
    return min(0.99, max(0.01, float(wp)))


# ============================================================
# Leverage Index
# ============================================================

# Representative plate appearance outcomes and their approximate probabilities
OUTCOME_PROBS = {
    "strikeout": 0.22,
    "groundout": 0.20,
    "flyout": 0.12,
    "single": 0.16,
    "walk": 0.09,
    "double": 0.05,
    "home_run": 0.03,
    "double_play": 0.03,
    "other_out": 0.10,
}

# Average WP swing per PA across all situations (empirical baseline)
LEAGUE_AVG_WP_SWING = 0.035


def _apply_outcome(runners: tuple[int, int, int], outs: int,
                    outcome: str) -> tuple[tuple[int, int, int], int, int]:
    """
    Apply a plate appearance outcome to a base-out state.
    Returns: (new_runners, new_outs, runs_scored)
    """
    r1, r2, r3 = runners

    if outcome == "strikeout":
        return (r1, r2, r3), min(outs + 1, 3), 0
    elif outcome == "groundout":
        return (0, r2, r3), min(outs + 1, 3), 0
    elif outcome == "flyout":
        # Sac fly: runner on 3rd scores with < 2 outs
        if r3 and outs < 2:
            return (r1, r2, 0), min(outs + 1, 3), 1
        return (r1, r2, r3), min(outs + 1, 3), 0
    elif outcome == "other_out":
        return (r1, r2, r3), min(outs + 1, 3), 0
    elif outcome == "single":
        runs = r3
        return (1, r1, r2), outs, runs
    elif outcome == "walk":
        if r1 and r2 and r3:
            return (1, 1, 1), outs, 1  # Bases loaded walk
        elif r1 and r2:
            return (1, 1, 1), outs, 0
        elif r1:
            return (1, 1, r3), outs, 0
        else:
            return (1, r2, r3), outs, 0
    elif outcome == "double":
        runs = r2 + r3
        return (0, 1, r1), outs, runs
    elif outcome == "home_run":
        runs = 1 + r1 + r2 + r3
        return (0, 0, 0), outs, runs
    elif outcome == "double_play":
        if r1 and outs < 2:
            return (0, r2, r3), min(outs + 2, 3), 0
        return (r1, r2, r3), min(outs + 1, 3), 0
    return (r1, r2, r3), outs, 0


def calculate_li(inning: int, top_bottom: str, outs: int,
                 runners: tuple[int, int, int], score_diff: int,
                 runs_per_game: float = MLB_RPG) -> float:
    """
    Calculate Leverage Index for the current situation.

    LI = Expected |WP change| for this PA / League average |WP change| per PA
    LI of 1.0 = average importance, >2.0 = high, >4.0 = very high
    """
    base_wp = calculate_wp(inning, top_bottom, outs, runners, score_diff, runs_per_game)

    total_wp_swing = 0.0
    for outcome, prob in OUTCOME_PROBS.items():
        new_runners, new_outs, runs = _apply_outcome(runners, outs, outcome)

        if new_outs >= 3:
            # Inning over - flip to next half
            if top_bottom == "top":
                new_wp = calculate_wp(inning, "bottom", 0, (0, 0, 0),
                                      score_diff - runs, runs_per_game)
            else:
                new_wp = calculate_wp(inning + 1, "top", 0, (0, 0, 0),
                                      score_diff + runs, runs_per_game)
        else:
            if top_bottom == "top":
                new_wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                      score_diff - runs, runs_per_game)
            else:
                new_wp = calculate_wp(inning, top_bottom, new_outs, new_runners,
                                      score_diff + runs, runs_per_game)

        total_wp_swing += prob * abs(new_wp - base_wp)

    li = total_wp_swing / LEAGUE_AVG_WP_SWING
    return round(li, 2)


def li_label(li: float) -> str:
    """Human-readable label for Leverage Index."""
    if li < 0.5:
        return "Low"
    elif li < 1.5:
        return "Medium"
    elif li < 3.0:
        return "High"
    else:
        return "Very High"


# ============================================================
# WPA (Win Probability Added) for a single play
# ============================================================

@dataclass
class GameState:
    inning: int
    top_bottom: str  # "top" or "bottom"
    outs: int
    runners: tuple[int, int, int]
    score_diff: int  # home - away
    runs_per_game: float = MLB_RPG


def calculate_wpa(before: GameState, after: GameState) -> dict:
    """Calculate WPA for a transition between two game states."""
    wp_before = calculate_wp(before.inning, before.top_bottom, before.outs,
                              before.runners, before.score_diff, before.runs_per_game)
    wp_after = calculate_wp(after.inning, after.top_bottom, after.outs,
                             after.runners, after.score_diff, after.runs_per_game)
    wpa = wp_after - wp_before
    return {
        "wpa": round(wpa, 4),
        "wp_before": round(wp_before, 4),
        "wp_after": round(wp_after, 4),
    }


# ============================================================
# Tactical Recommendation Engine (RE24-based)
# ============================================================

TACTICS = {
    "sacrifice_bunt": {
        "name": "Sacrifice Bunt",
        "name_ja": "送りバント",
        "success_rate": 0.80,
        "requires_runner_1b": True,
        "max_outs": 1,  # Only with 0 or 1 out
    },
    "steal_2b": {
        "name": "Steal 2nd Base",
        "name_ja": "盗塁（二塁）",
        "success_rate": 0.72,
        "requires_runner_1b": True,
    },
    "steal_3b": {
        "name": "Steal 3rd Base",
        "name_ja": "盗塁（三塁）",
        "success_rate": 0.65,
        "requires_runner_2b": True,
    },
    "intentional_walk": {
        "name": "Intentional Walk",
        "name_ja": "敬遠",
        "requires_open_1b": True,
    },
    "pitching_change": {
        "name": "Pitching Change",
        "name_ja": "継投",
    },
    "pinch_hitter": {
        "name": "Pinch Hitter",
        "name_ja": "代打",
    },
    "hit_and_run": {
        "name": "Hit and Run",
        "name_ja": "エンドラン",
        "success_rate": 0.55,
        "requires_runner_1b": True,
    },
    "squeeze_play": {
        "name": "Squeeze Play",
        "name_ja": "スクイズ",
        "success_rate": 0.60,
        "requires_runner_3b": True,
        "max_outs": 1,
    },
}


def _evaluate_tactic(tactic_key: str, tactic: dict,
                     runners: tuple[int, int, int], outs: int,
                     score_diff: int, inning: int, top_bottom: str,
                     runs_per_game: float) -> dict | None:
    """Evaluate a single tactic and return recommendation if applicable."""
    r1, r2, r3 = runners

    # Check preconditions
    if tactic.get("requires_runner_1b") and not r1:
        return None
    if tactic.get("requires_runner_2b") and not r2:
        return None
    if tactic.get("requires_runner_3b") and not r3:
        return None
    if tactic.get("requires_open_1b") and r1:
        return None
    if "max_outs" in tactic and outs > tactic["max_outs"]:
        return None

    current_re = get_re24(runners, outs, runs_per_game)
    success_rate = tactic.get("success_rate")

    if tactic_key == "sacrifice_bunt":
        # Success: runner advances, batter out
        new_runners_s = (0, r2 or r1, r3 or (r2 if r1 else 0))
        runs_s = 1 if r3 and r1 else 0
        re_success = get_re24(new_runners_s, outs + 1, runs_per_game) + runs_s
        # Failure: batter out, runners stay
        re_fail = get_re24(runners, outs + 1, runs_per_game)
        ev = success_rate * re_success + (1 - success_rate) * re_fail
        delta = ev - current_re

    elif tactic_key == "steal_2b":
        new_runners_s = (0, 1, r3)
        re_success = get_re24(new_runners_s, outs, runs_per_game)
        re_fail = get_re24((0, r2, r3), outs + 1, runs_per_game)
        ev = success_rate * re_success + (1 - success_rate) * re_fail
        delta = ev - current_re

    elif tactic_key == "steal_3b":
        new_runners_s = (r1, 0, 1)
        re_success = get_re24(new_runners_s, outs, runs_per_game)
        re_fail = get_re24((r1, 0, r3), outs + 1, runs_per_game)
        ev = success_rate * re_success + (1 - success_rate) * re_fail
        delta = ev - current_re

    elif tactic_key == "intentional_walk":
        new_runners = (1, r2, r3)
        re_after = get_re24(new_runners, outs, runs_per_game)
        delta = -(re_after - current_re)  # Negative for defense perspective

    elif tactic_key == "hit_and_run":
        # Success: single + runner advances extra base
        re_success = get_re24((1, 0, 1 if r1 else r3), outs, runs_per_game) + (1 if r3 else 0)
        # Failure: lineout/groundout, runner exposed
        re_fail = get_re24((0, r2, r3), outs + 1, runs_per_game)
        ev = success_rate * re_success + (1 - success_rate) * re_fail
        delta = ev - current_re

    elif tactic_key == "squeeze_play":
        re_success = get_re24((r1 if not r1 else 0, r2, 0), outs + 1, runs_per_game) + 1
        re_fail = get_re24((r1, r2, 0), outs + 1, runs_per_game)
        ev = success_rate * re_success + (1 - success_rate) * re_fail
        delta = ev - current_re

    elif tactic_key in ("pitching_change", "pinch_hitter"):
        # Situational - always suggest in high leverage
        li = calculate_li(inning, top_bottom, outs, runners, score_diff, runs_per_game)
        if li < 1.5:
            return None
        return {
            "tactic": tactic["name"],
            "tactic_ja": tactic["name_ja"],
            "reason": f"High leverage situation (LI={li:.1f})",
            "re24_delta": 0.0,
            "recommendation": "Consider",
        }
    else:
        return None

    rec = "Recommended" if delta > 0.02 else "Neutral" if delta > -0.02 else "Not recommended"

    return {
        "tactic": tactic["name"],
        "tactic_ja": tactic["name_ja"],
        "re24_delta": round(delta, 3),
        "recommendation": rec,
        "success_rate": success_rate,
    }


def get_tactical_recommendations(inning: int, top_bottom: str, outs: int,
                                  runners: tuple[int, int, int], score_diff: int,
                                  runs_per_game: float = MLB_RPG) -> list[dict]:
    """Get all applicable tactical recommendations for the current situation."""
    recs = []
    for key, tactic in TACTICS.items():
        result = _evaluate_tactic(key, tactic, runners, outs, score_diff,
                                   inning, top_bottom, runs_per_game)
        if result:
            recs.append(result)
    # Sort: recommended first, then by RE24 delta descending
    order = {"Recommended": 0, "Consider": 1, "Neutral": 2, "Not recommended": 3}
    recs.sort(key=lambda x: (order.get(x["recommendation"], 9), -x["re24_delta"]))
    return recs


# ============================================================
# Matchup Quality Adjustment
# ============================================================


def adjust_wp_for_matchup(base_wp: float, batter_ops: float | None = None,
                           pitcher_era: float | None = None) -> float:
    """
    Adjust WP based on batter/pitcher quality.
    Uses logit-space adjustment for smooth behavior near 0/1.
    """
    if batter_ops is None and pitcher_era is None:
        return base_wp

    # Convert to logit space
    p = max(0.01, min(0.99, base_wp))
    logit = np.log(p / (1 - p))

    # Batter adjustment: OPS above .750 = positive, below = negative
    if batter_ops is not None:
        batter_adj = (batter_ops - 0.750) * 0.5
        logit += batter_adj

    # Pitcher adjustment: ERA below 3.50 = positive (for pitcher's team), above = negative
    if pitcher_era is not None:
        pitcher_adj = (3.50 - pitcher_era) * 0.15
        logit -= pitcher_adj  # Negative because good pitcher hurts batter's team

    # Convert back
    wp = 1.0 / (1.0 + np.exp(-logit))
    return round(min(0.99, max(0.01, float(wp))), 4)


# ============================================================
# Preset Scenarios
# ============================================================

SCENARIOS = {
    "ninth_inning_drama": {
        "name": "9th Inning Drama",
        "name_ja": "9回裏2アウト満塁同点",
        "description": "Bottom 9th, 2 outs, bases loaded, tie game",
        "state": GameState(9, "bottom", 2, (1, 1, 1), 0),
    },
    "game_start": {
        "name": "Game Start",
        "name_ja": "1回表開始",
        "description": "Top of 1st, no outs, bases empty, 0-0",
        "state": GameState(1, "top", 0, (0, 0, 0), 0),
    },
    "rally_7th": {
        "name": "7th Inning Rally",
        "name_ja": "7回裏1点ビハインド1死1-2塁",
        "description": "Bottom 7th, 1 out, runners on 1st & 2nd, down by 1",
        "state": GameState(7, "bottom", 1, (1, 1, 0), -1),
    },
    "tied_8th": {
        "name": "Tied 8th",
        "name_ja": "8回表同点無死走者なし",
        "description": "Top of 8th, no outs, bases empty, tie game",
        "state": GameState(8, "top", 0, (0, 0, 0), 0),
    },
    "walkoff_chance": {
        "name": "Walk-off Chance",
        "name_ja": "9回裏サヨナラ1死2-3塁",
        "description": "Bottom 9th, 1 out, runners on 2nd & 3rd, tie game",
        "state": GameState(9, "bottom", 1, (0, 1, 1), 0),
    },
    "comfortable_lead": {
        "name": "Comfortable Lead",
        "name_ja": "5回表3点リード",
        "description": "Top of 5th, no outs, bases empty, home team up by 3",
        "state": GameState(5, "top", 0, (0, 0, 0), 3),
    },
}


def analyze_scenario(scenario_name: str, runs_per_game: float = MLB_RPG) -> dict:
    """Analyze a preset scenario and return full WP/LI/tactics."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        return {"error": f"Unknown scenario: {scenario_name}"}

    s = scenario["state"]
    wp = calculate_wp(s.inning, s.top_bottom, s.outs, s.runners, s.score_diff, runs_per_game)
    li = calculate_li(s.inning, s.top_bottom, s.outs, s.runners, s.score_diff, runs_per_game)
    tactics = get_tactical_recommendations(s.inning, s.top_bottom, s.outs, s.runners,
                                            s.score_diff, runs_per_game)

    return {
        "scenario": scenario["name"],
        "scenario_ja": scenario["name_ja"],
        "description": scenario["description"],
        "game_state": {
            "inning": s.inning,
            "top_bottom": s.top_bottom,
            "outs": s.outs,
            "runners": {"1B": s.runners[0], "2B": s.runners[1], "3B": s.runners[2]},
            "score_diff": s.score_diff,
        },
        "win_probability": round(wp, 4),
        "win_probability_pct": f"{wp * 100:.1f}%",
        "leverage_index": li,
        "leverage_label": li_label(li),
        "tactics": tactics,
    }


# ============================================================
# Convenience: Full analysis for a given game state
# ============================================================


def full_analysis(inning: int, top_bottom: str, outs: int,
                  runners: tuple[int, int, int], score_diff: int,
                  runs_per_game: float = MLB_RPG,
                  batter_ops: float | None = None,
                  pitcher_era: float | None = None) -> dict:
    """Full analysis: WP + LI + tactics + matchup adjustment."""
    wp = calculate_wp(inning, top_bottom, outs, runners, score_diff, runs_per_game)
    li = calculate_li(inning, top_bottom, outs, runners, score_diff, runs_per_game)
    tactics = get_tactical_recommendations(inning, top_bottom, outs, runners,
                                            score_diff, runs_per_game)

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
        "adjusted_wp": round(adjusted_wp, 4) if (batter_ops or pitcher_era) else None,
        "leverage_index": li,
        "leverage_label": li_label(li),
        "tactics": tactics,
    }
