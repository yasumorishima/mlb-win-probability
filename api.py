"""
MLB Win Probability API

Win probability, Leverage Index, run expectancy and tactics from a game
state, computed by the engine fitted to real plate-appearance transitions
(wp_engine.py, engine_tables.json) through analysis.py. /wp/advice asks
Claude (strategist.py) for a recommendation; it needs ANTHROPIC_API_KEY.
"""

from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from analysis import (
    SCENARIOS,
    GameState,
    analyze_scenario,
    engine_info,
    full_analysis,
    get_advice,
    has_api_key,
    re24_table,
    wpa,
)
from live_feed import get_live_wp, get_todays_games

app = FastAPI(
    title="MLB Win Probability API",
    description=(
        "MLB game state to win probability, Leverage Index, run expectancy and tactics.\n\n"
        "## Engine\n"
        "- **Win probability**: plate-appearance transitions of MLB regular-season games "
        "(Stats API play-by-play), run distributions per half-inning by dynamic programming, "
        "then a recursion over half-innings with walk-off rules and the extra-inning runner\n"
        "- **Count**: optional `balls`/`strikes` shift the mix of plate-appearance outcomes\n"
        "- **Leverage Index**: expected |WP change| of the next plate appearance over the league average\n"
        "- **Tactics**: stolen-base break-even success rate; sacrifice bunt and intentional walk "
        "as a change in win probability for the side that chooses them\n"
        "- Every player is treated as league-average.\n\n"
        "## Claude bench coach\n"
        "`/wp/advice` sends the situation and the engine's numbers to Claude and returns its "
        "recommendation with the numbers it cited (503 when ANTHROPIC_API_KEY is not set)."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Lang(str, Enum):
    ja = "JA"
    en = "EN"


class TopBottom(str, Enum):
    top = "top"
    bottom = "bottom"


ScenarioName = Enum("ScenarioName", {k: k for k in SCENARIOS}, type=str)


def _analysis(inning, top_bottom, outs, runner1, runner2, runner3, score_diff,
              scheduled_innings, balls, strikes):
    if (balls is None) != (strikes is None):
        raise HTTPException(status_code=422, detail="give both balls and strikes, or neither")
    try:
        return full_analysis(inning, top_bottom.value, outs, (runner1, runner2, runner3),
                             score_diff, scheduled_innings, balls, strikes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None




@app.get("/")
def root():
    return {
        "name": "MLB Win Probability API",
        "version": app.version,
        "model": engine_info(),
        "endpoints": [
            "/wp",
            "/wp/advice",
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
        "Home win probability for the given state, the batting team's win probability, "
        "Leverage Index, run expectancy for the rest of the half, WP after each kind of "
        "plate-appearance outcome, and tactics."
    ),
)
def get_wp(
    inning: int = Query(ge=1, le=20, description="Current inning"),
    top_bottom: TopBottom = Query(description="Top or bottom of inning"),
    outs: int = Query(ge=0, le=2, description="Number of outs (0-2)"),
    runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1st base (0 or 1)"),
    runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2nd base (0 or 1)"),
    runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3rd base (0 or 1)"),
    score_diff: int = Query(default=0, ge=-30, le=30, description="Score difference (home - away)"),
    scheduled_innings: int = Query(default=9, ge=1, le=9, description="Scheduled innings (9; 7 for 2020-21 doubleheaders)"),
    balls: int | None = Query(default=None, ge=0, le=3, description="Balls in the current count (optional)"),
    strikes: int | None = Query(default=None, ge=0, le=2, description="Strikes in the current count (optional)"),
):
    return _analysis(inning, top_bottom, outs, runner1, runner2, runner3, score_diff,
                     scheduled_innings, balls, strikes)


@app.get(
    "/wp/advice",
    summary="Claude bench coach",
    description=(
        "Claude's recommendation for the next pitch, from the situation and the engine's numbers "
        "(the same analysis as /wp). Returns the recommendation, its reason, the numbers it cited "
        "and any cited number that does not match the input. "
        "Requires the `ANTHROPIC_API_KEY` environment variable (503 otherwise)."
    ),
)
def get_wp_advice(
    inning: int = Query(ge=1, le=20, description="Current inning"),
    top_bottom: TopBottom = Query(description="Top or bottom of inning"),
    outs: int = Query(ge=0, le=2, description="Number of outs (0-2)"),
    runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1st base (0 or 1)"),
    runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2nd base (0 or 1)"),
    runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3rd base (0 or 1)"),
    score_diff: int = Query(default=0, ge=-30, le=30, description="Score difference (home - away)"),
    scheduled_innings: int = Query(default=9, ge=1, le=9, description="Scheduled innings"),
    balls: int | None = Query(default=None, ge=0, le=3, description="Balls (optional)"),
    strikes: int | None = Query(default=None, ge=0, le=2, description="Strikes (optional)"),
    lang: Lang = Query(default=Lang.en, description="Language of the reason (EN or JA)"),
):
    if not has_api_key():
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set. Set it in the server environment to enable /wp/advice.",
        )
    result = _analysis(inning, top_bottom, outs, runner1, runner2, runner3, score_diff,
                       scheduled_innings, balls, strikes)
    if result["game_over"]:
        raise HTTPException(status_code=422, detail="the game is over in this state")
    adv, problems, payload = get_advice(result, lang.value)
    if adv is None:
        raise HTTPException(status_code=502, detail="no valid response from the model")
    return {
        **result,
        "advice": adv.model_dump(),
        "citation_problems": problems,
        "advice_input": payload,
    }


@app.get(
    "/wp/play",
    summary="WPA (Win Probability Added) for a single play",
    description=(
        "Change in home win probability between two states. `after_outs=3` means the play "
        "ended the half-inning."
    ),
)
def get_wpa(
    before_inning: int = Query(ge=1, le=20, description="Inning before play"),
    before_top_bottom: TopBottom = Query(description="Top/bottom before play"),
    before_outs: int = Query(ge=0, le=2, description="Outs before play"),
    before_runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1B before"),
    before_runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2B before"),
    before_runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3B before"),
    before_score_diff: int = Query(default=0, ge=-30, le=30, description="Score diff before"),
    after_inning: int = Query(ge=1, le=20, description="Inning after play"),
    after_top_bottom: TopBottom = Query(description="Top/bottom after play"),
    after_outs: int = Query(ge=0, le=3, description="Outs after play (0-3)"),
    after_runner1: int = Query(default=0, ge=0, le=1, description="Runner on 1B after"),
    after_runner2: int = Query(default=0, ge=0, le=1, description="Runner on 2B after"),
    after_runner3: int = Query(default=0, ge=0, le=1, description="Runner on 3B after"),
    after_score_diff: int = Query(default=0, ge=-30, le=30, description="Score diff after"),
    scheduled_innings: int = Query(default=9, ge=1, le=9, description="Scheduled innings"),
):
    before = GameState(before_inning, before_top_bottom.value, before_outs,
                       (before_runner1, before_runner2, before_runner3),
                       before_score_diff, scheduled_innings)
    after = GameState(after_inning, after_top_bottom.value, after_outs,
                      (after_runner1, after_runner2, after_runner3),
                      after_score_diff, scheduled_innings)
    try:
        result = wpa(before, after)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    for key, s in (("before_state", before), ("after_state", after)):
        result[key] = {
            "inning": s.inning, "top_bottom": s.top_bottom, "outs": s.outs,
            "runners": {"1B": s.runners[0], "2B": s.runners[1], "3B": s.runners[2]},
            "score_diff": s.score_diff,
        }
    return result


@app.get(
    "/re24",
    summary="Run expectancy for the 24 base-out states",
    description=(
        "Expected runs in the rest of the half-inning and P(at least one run) for each "
        "base-out state, from the engine's run distributions (top = visitors batting, "
        "bottom = home batting)."
    ),
)
def get_re24(half: TopBottom = Query(default=TopBottom.top, description="Half-inning")):
    table = re24_table(half.value)
    return {"half": half.value, "count": len(table), "re24_table": table}


@app.get(
    "/wp/scenario",
    summary="Preset Scenario Analysis",
    description="Analyze a preset game scenario (e.g., 9th inning drama, walk-off chance).",
)
def get_scenario(name: ScenarioName = Query(description="Scenario name")):
    return analyze_scenario(name.value)


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
        "Fetch the live game state from the MLB Stats API and compute win probability, "
        "Leverage Index and tactics.\n\nUse `/games/today` to get valid `gamePk` values."
    ),
)
def wp_live(game_pk: int):
    result = get_live_wp(game_pk)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Game {game_pk} not found or API unavailable")
    return result
