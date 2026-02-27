"""
MLB Live Feed

Fetch today's games and live game state from the MLB Stats API.
No external dependencies — uses stdlib urllib.request + json only.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

_BASE = "https://statsapi.mlb.com/api"


def _fetch(url: str) -> dict | None:
    """GET JSON from url. Returns None on any network or HTTP error."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def get_todays_games(date: str | None = None) -> list[dict]:
    """
    Return today's MLB schedule.

    Args:
        date: "YYYY-MM-DD" (default: today UTC)

    Returns:
        List of dicts with keys:
            gamePk, away_team, home_team, status, start_time_utc
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    data = _fetch(f"{_BASE}/v1/schedule?date={date}&sportId=1")
    if data is None:
        return []

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            games.append({
                "gamePk": g["gamePk"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_team": g["teams"]["home"]["team"]["name"],
                "status": g["status"]["detailedState"],
                "start_time_utc": g.get("gameDate", ""),
            })
    return games


def get_live_state(game_pk: int) -> dict | None:
    """
    Fetch live game state for a single game.

    Returns dict with keys:
        inning, top_bottom, outs, runners (tuple[int,int,int]),
        score_home, score_away, score_diff,
        batter_name, pitcher_name,
        away_team, home_team, status
    Returns None if game not found or API error.
    """
    data = _fetch(f"{_BASE}/v1.1/game/{game_pk}/feed/live")
    if data is None:
        return None

    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})
    linescore = live_data.get("linescore", {})

    status = game_data.get("status", {}).get("detailedState", "Unknown")

    # Teams
    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("name", "Away")
    home_team = teams.get("home", {}).get("name", "Home")

    # Scores
    ls_teams = linescore.get("teams", {})
    score_home = ls_teams.get("home", {}).get("runs", 0)
    score_away = ls_teams.get("away", {}).get("runs", 0)
    score_diff = score_home - score_away

    # Inning / half / outs
    inning = linescore.get("currentInning", 1)
    is_top = linescore.get("isTopInning", True)
    top_bottom = "top" if is_top else "bottom"
    outs = linescore.get("outs", 0)
    # outs=3 means side is over — clamp to 2 for WP engine
    outs = min(outs, 2)

    # Runners
    offense = linescore.get("offense", {})
    runner1 = 1 if offense.get("first") else 0
    runner2 = 1 if offense.get("second") else 0
    runner3 = 1 if offense.get("third") else 0
    runners = (runner1, runner2, runner3)

    # Player names
    batter_name = offense.get("batter", {}).get("fullName", "")
    defense = linescore.get("defense", {})
    pitcher_name = defense.get("pitcher", {}).get("fullName", "")

    return {
        "inning": inning,
        "top_bottom": top_bottom,
        "outs": outs,
        "runners": runners,
        "score_home": score_home,
        "score_away": score_away,
        "score_diff": score_diff,
        "batter_name": batter_name,
        "pitcher_name": pitcher_name,
        "away_team": away_team,
        "home_team": home_team,
        "status": status,
    }


def get_live_wp(game_pk: int, runs_per_game: float = 4.5) -> dict | None:
    """
    Fetch live game state and compute Win Probability + full analysis.

    Returns combined dict: live state + full_analysis() output.
    Returns None if game not found or not in a computable state.
    """
    from win_probability import full_analysis

    state = get_live_state(game_pk)
    if state is None:
        return None

    # If game hasn't started or is already final, return state + flag
    if state["status"] in ("Scheduled", "Pre-Game", "Warmup"):
        return {**state, "status_note": "Game not yet started", "win_probability": None}

    analysis = full_analysis(
        state["inning"],
        state["top_bottom"],
        state["outs"],
        state["runners"],
        state["score_diff"],
        runs_per_game,
    )
    return {**state, **analysis}
