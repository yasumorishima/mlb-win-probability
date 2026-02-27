"""
MLB Win Probability API

Calculate real-time Win Probability, Leverage Index, and tactical recommendations
from game state inputs. Markov Chain + RE24 approach, no external data dependencies.
"""

from enum import Enum

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from win_probability import (
    MLB_RPG,
    SCENARIOS,
    GameState,
    analyze_scenario,
    calculate_li,
    calculate_wp,
    calculate_wpa,
    full_analysis,
    get_re24_table,
    li_label,
)
from live_feed import get_todays_games, get_live_state, get_live_wp

app = FastAPI(
    title="MLB Win Probability API",
    description=(
        "MLB/NPB game state to Win Probability, Leverage Index, and tactical recommendations.\n\n"
        "## Approach\n"
        "- **Win Probability**: Markov Chain + Normal distribution approximation\n"
        "- **RE24**: Run Expectancy by 24 base-out states (MLB 2010-2019 avg)\n"
        "- **Leverage Index**: Expected WP swing relative to league average\n"
        "- **Tactics**: RE24-based expected value comparison of 8 tactical options\n\n"
        "## Scoring Environment\n"
        "- MLB default: 4.5 runs/game\n"
        "- NPB: 4.0 runs/game (pass `runs_per_game=4.0`)\n"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopBottom(str, Enum):
    top = "top"
    bottom = "bottom"


class ScenarioName(str, Enum):
    ninth_inning_drama = "ninth_inning_drama"
    game_start = "game_start"
    rally_7th = "rally_7th"
    tied_8th = "tied_8th"
    walkoff_chance = "walkoff_chance"
    comfortable_lead = "comfortable_lead"


@app.get("/")
def root():
    return {
        "name": "MLB Win Probability API",
        "version": "0.2.0",
        "endpoints": [
            "/wp",
            "/wp/play",
            "/re24",
            "/wp/scenario",
            "/games/today",
            "/wp/live/{gamePk}",
        ],
        "scenarios": list(SCENARIOS.keys()),
    }


@app.get(
    "/wp",
    summary="Win Probability + Leverage Index + Tactics",
    description=(
        "Calculate Win Probability for the home team given the current game state. "
        "Also returns Leverage Index and tactical recommendations."
    ),
)
def get_wp(
    inning: int = Query(ge=1, le=15, description="Current inning (1-15)", examples=[1, 7, 9]),
    top_bottom: TopBottom = Query(description="Top or bottom of inning"),
    outs: int = Query(ge=0, le=2, description="Number of outs (0-2)"),
    runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1st base (0 or 1)"),
    runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2nd base (0 or 1)"),
    runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3rd base (0 or 1)"),
    score_diff: int = Query(default=0, ge=-20, le=20, description="Score difference (home - away)"),
    runs_per_game: float = Query(default=MLB_RPG, ge=2.0, le=8.0, description="Scoring environment (MLB=4.5, NPB=4.0)"),
    batter_ops: float | None = Query(default=None, ge=0.0, le=2.0, description="Batter OPS for matchup adjustment"),
    pitcher_era: float | None = Query(default=None, ge=0.0, le=15.0, description="Pitcher ERA for matchup adjustment"),
):
    runners = (runner1, runner2, runner3)
    result = full_analysis(
        inning, top_bottom.value, outs, runners, score_diff,
        runs_per_game, batter_ops, pitcher_era,
    )
    return result


@app.get(
    "/wp/play",
    summary="WPA (Win Probability Added) for a single play",
    description=(
        "Calculate the WPA for a transition from one game state to another. "
        "Provide before and after states."
    ),
)
def get_wpa(
    # Before state
    before_inning: int = Query(ge=1, le=15, description="Inning before play"),
    before_top_bottom: TopBottom = Query(description="Top/bottom before play"),
    before_outs: int = Query(ge=0, le=2, description="Outs before play"),
    before_runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1B before"),
    before_runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2B before"),
    before_runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3B before"),
    before_score_diff: int = Query(default=0, ge=-20, le=20, description="Score diff before"),
    # After state
    after_inning: int = Query(ge=1, le=15, description="Inning after play"),
    after_top_bottom: TopBottom = Query(description="Top/bottom after play"),
    after_outs: int = Query(ge=0, le=3, description="Outs after play (0-3)"),
    after_runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1B after"),
    after_runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2B after"),
    after_runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3B after"),
    after_score_diff: int = Query(default=0, ge=-20, le=20, description="Score diff after"),
    runs_per_game: float = Query(default=MLB_RPG, ge=2.0, le=8.0, description="Scoring environment"),
):
    before = GameState(
        before_inning, before_top_bottom.value, before_outs,
        (before_runner1, before_runner2, before_runner3),
        before_score_diff, runs_per_game,
    )
    after = GameState(
        after_inning, after_top_bottom.value, min(after_outs, 2),
        (after_runner1, after_runner2, after_runner3),
        after_score_diff, runs_per_game,
    )
    result = calculate_wpa(before, after)
    result["before_state"] = {
        "inning": before.inning, "top_bottom": before.top_bottom,
        "outs": before.outs,
        "runners": {"1B": before.runners[0], "2B": before.runners[1], "3B": before.runners[2]},
        "score_diff": before.score_diff,
    }
    result["after_state"] = {
        "inning": after.inning, "top_bottom": after.top_bottom,
        "outs": after.outs,
        "runners": {"1B": after.runners[0], "2B": after.runners[1], "3B": after.runners[2]},
        "score_diff": after.score_diff,
    }
    return result


@app.get(
    "/re24",
    summary="RE24 Table (Run Expectancy by 24 base-out states)",
    description="Returns the full RE24 table scaled by the given scoring environment.",
)
def get_re24(
    runs_per_game: float = Query(default=MLB_RPG, ge=2.0, le=8.0, description="Scoring environment (MLB=4.5, NPB=4.0)"),
):
    table = get_re24_table(runs_per_game)
    return {
        "runs_per_game": runs_per_game,
        "count": len(table),
        "re24_table": table,
    }


@app.get(
    "/wp/scenario",
    summary="Preset Scenario Analysis",
    description="Analyze a preset game scenario (e.g., 9th inning drama, walk-off chance).",
)
def get_scenario(
    name: ScenarioName = Query(description="Scenario name"),
    runs_per_game: float = Query(default=MLB_RPG, ge=2.0, le=8.0, description="Scoring environment"),
):
    result = analyze_scenario(name.value, runs_per_game)
    return result


@app.get(
    "/games/today",
    summary="Today's MLB Schedule",
    description="Return today's MLB games with status (Scheduled / In Progress / Final).",
)
def games_today(
    date: str | None = Query(default=None, description="Date in YYYY-MM-DD (default: today UTC)"),
):
    games = get_todays_games(date)
    return {"date": date or "today", "count": len(games), "games": games}


@app.get(
    "/wp/live/{game_pk}",
    summary="Live Win Probability for a Game",
    description=(
        "Fetch live game state from MLB Stats API and compute Win Probability + LI + tactics.\n\n"
        "Use `/games/today` to get valid `gamePk` values."
    ),
)
def wp_live(
    game_pk: int,
    runs_per_game: float = Query(default=MLB_RPG, ge=2.0, le=8.0, description="Scoring environment (MLB=4.5, NPB=4.0)"),
):
    from fastapi import HTTPException
    result = get_live_wp(game_pk, runs_per_game)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Game {game_pk} not found or API unavailable")
    return result
