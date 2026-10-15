"""
MLB Live Feed

Fetch today's games and live game state from the MLB Stats API (keyless).
No external dependencies — uses stdlib urllib.request + json only.

Game states follow scripts/build_states.py, which the engine was fitted on:
the state before a play is the previous play's count.outs (0 at the start
of a half) and matchup.postOnFirst/Second/Third; regular-season (and
spring) games from 2020 start every extra half-inning with a runner on
second.
"""

import json
import http.client
import urllib.request
import urllib.error
from datetime import datetime, timezone

_BASE = "https://statsapi.mlb.com/api"

# Game types that start extra innings with a runner on second (2020+).
# Postseason games (F, D, L, W) do not.
AUTO_RUNNER_GAME_TYPES = ("R", "S")


def _fetch(url: str) -> dict | None:
    """GET JSON from url. Returns None on any network, HTTP or decoding error
    (ValueError covers JSONDecodeError and UnicodeDecodeError; HTTPException
    covers InvalidURL and IncompleteRead)."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, http.client.HTTPException, ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


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
            if "gamePk" not in g:
                continue
            teams = g.get("teams", {})
            games.append({
                "gamePk": g["gamePk"],
                "away_team": teams.get("away", {}).get("team", {}).get("name", "Away"),
                "home_team": teams.get("home", {}).get("team", {}).get("name", "Home"),
                "status": g.get("status", {}).get("detailedState", "Unknown"),
                "start_time_utc": g.get("gameDate", ""),
            })
    return games


GAME_OVER_PREFIXES = ("Final", "Game Over", "Completed Early")
NO_STATE_PREFIXES = GAME_OVER_PREFIXES + ("Postponed", "Cancelled", "Suspended")


def is_game_over_status(status: str) -> bool:
    return status.startswith(GAME_OVER_PREFIXES)


def has_live_state(status: str) -> bool:
    """False when the game is over or not being played, so its state is not
    something to analyse."""
    return not status.startswith(NO_STATE_PREFIXES)


def _game_rules(data: dict) -> tuple[int, bool]:
    """(scheduled innings, automatic runner on second in extra innings)."""
    game = data.get("gameData", {}).get("game", {})
    try:
        season = int(game.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    sched = int(data.get("liveData", {}).get("linescore", {}).get("scheduledInnings") or 9)
    return sched, season >= 2020 and game.get("type") in AUTO_RUNNER_GAME_TYPES


def _game_ids(data: dict) -> dict:
    """Team ids, official date and game type, for the team layer (team_layer.py)."""
    game_data = data.get("gameData", {})
    teams = game_data.get("teams", {})
    return {"home_id": teams.get("home", {}).get("id"),
            "away_id": teams.get("away", {}).get("id"),
            "game_date": game_data.get("datetime", {}).get("officialDate"),
            "game_type": game_data.get("game", {}).get("type")}


def _half_start(inning: int, sched: int, extra_runner: bool) -> tuple[int, int, int]:
    return (0, 1, 0) if extra_runner and inning > sched else (0, 0, 0)


def _bases_after(play: dict) -> tuple[int, int, int]:
    m = play.get("matchup", {})
    return tuple(1 if m.get(k) else 0 for k in ("postOnFirst", "postOnSecond", "postOnThird"))


def _pitch_hit(event: dict) -> tuple[dict | None, dict | None]:
    pitch_data = hit_data = None
    pd_raw = event.get("pitchData")
    if pd_raw:
        coords = pd_raw.get("coordinates", {})
        pitch_data = {
            "startSpeed": pd_raw.get("startSpeed"),
            "endSpeed": pd_raw.get("endSpeed"),
            "spinRate": pd_raw.get("spinRate"),
            "zone": pd_raw.get("zone"),
            "extension": pd_raw.get("extension"),
            "coordinates": {
                "pfxX": coords.get("pfxX"),
                "pfxZ": coords.get("pfxZ"),
                "pX": coords.get("pX"),
                "pZ": coords.get("pZ"),
            },
        }
    hd_raw = event.get("hitData")
    if hd_raw:
        hit_data = {
            "launchSpeed": hd_raw.get("launchSpeed"),
            "launchAngle": hd_raw.get("launchAngle"),
            "totalDistance": hd_raw.get("totalDistance"),
        }
    return pitch_data, hit_data


def parse_live_state(data: dict) -> dict:
    """Current game state from a /feed/live response (see get_live_state)."""
    game_data = data.get("gameData", {})
    live_data = data.get("liveData", {})
    linescore = live_data.get("linescore", {})
    sched, extra_runner = _game_rules(data)

    status = game_data.get("status", {}).get("detailedState", "Unknown")
    teams = game_data.get("teams", {})
    away_team = teams.get("away", {}).get("name", "Away")
    home_team = teams.get("home", {}).get("name", "Home")

    ls_teams = linescore.get("teams", {})
    score_home = ls_teams.get("home", {}).get("runs", 0)
    score_away = ls_teams.get("away", {}).get("runs", 0)

    offense = linescore.get("offense", {})
    current_play = live_data.get("plays", {}).get("currentPlay", {})
    about = current_play.get("about", {})
    count = current_play.get("count", {})

    # The state comes from the current play, as in scripts/build_states.py:
    # a plate appearance in progress -> its half, its outs so far and the
    # runners on base now; a finished play -> the state it left behind.
    if about:
        inning = about.get("inning", 1)
        top_bottom = "top" if about.get("isTopInning", True) else "bottom"
    else:
        inning = linescore.get("currentInning", 1)
        top_bottom = "top" if linescore.get("isTopInning", True) else "bottom"
    outs = count.get("outs", linescore.get("outs", 0)) or 0
    if about.get("isComplete", False):
        runners = _bases_after(current_play)
        balls = strikes = 0
    else:
        runners = tuple(1 if offense.get(k) else 0 for k in ("first", "second", "third"))
        balls, strikes = count.get("balls", 0), count.get("strikes", 0)

    # Three outs: the next state is the start of the next half, not
    # "two outs" of the half that just ended. A finished game has no next
    # half, so it keeps the state its last play left.
    if outs >= 3 and not is_game_over_status(status):
        if top_bottom == "top":
            top_bottom = "bottom"
        else:
            top_bottom, inning = "top", inning + 1
        outs, runners = 0, _half_start(inning, sched, extra_runner)
        balls = strikes = 0

    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
        balls = strikes = None  # the plate appearance has ended

    pitch_data = hit_data = None
    play_events = current_play.get("playEvents", [])
    if play_events:
        pitch_data, hit_data = _pitch_hit(play_events[-1])

    return {
        "inning": inning,
        "top_bottom": top_bottom,
        "outs": outs,
        "runners": runners,
        "score_home": score_home,
        "score_away": score_away,
        "score_diff": score_home - score_away,
        "scheduled_innings": sched,
        "extra_runner": extra_runner,
        "batter_name": offense.get("batter", {}).get("fullName", ""),
        "pitcher_name": linescore.get("defense", {}).get("pitcher", {}).get("fullName", ""),
        "away_team": away_team,
        "home_team": home_team,
        "status": status,
        "balls": balls,
        "strikes": strikes,
        "pitch_data": pitch_data,
        "hit_data": hit_data,
        **_game_ids(data),
    }


def get_live_state(game_pk: int) -> dict | None:
    """
    Fetch live game state for a single game.

    Returns dict with keys:
        inning, top_bottom, outs, runners (tuple[int,int,int]),
        score_home, score_away, score_diff, scheduled_innings, extra_runner,
        batter_name, pitcher_name, away_team, home_team, status,
        balls, strikes (None between plate appearances), pitch_data, hit_data
    Returns None if game not found or API error.
    """
    data = _fetch(f"{_BASE}/v1.1/game/{game_pk}/feed/live")
    if data is None:
        return None
    return parse_live_state(data)


def parse_game_plays(data: dict) -> list[dict]:
    """Completed plays of a /feed/live response with the state before and
    after each one (see get_game_plays)."""
    teams = data.get("gameData", {}).get("teams", {})
    home_team = teams.get("home", {}).get("name", "Home")
    away_team = teams.get("away", {}).get("name", "Away")
    sched, extra_runner = _game_rules(data)
    ids = _game_ids(data)

    plays = []
    prev_half = None
    outs, bases = 0, (0, 0, 0)
    prev_home = prev_away = 0

    for play in data.get("liveData", {}).get("plays", {}).get("allPlays", []):
        about = play.get("about", {})
        if not about.get("isComplete", False):
            continue
        result = play.get("result", {})
        matchup = play.get("matchup", {})
        inning = about.get("inning", 1)
        top_bottom = "top" if about.get("isTopInning", True) else "bottom"
        if (inning, top_bottom) != prev_half:
            outs, bases = 0, _half_start(inning, sched, extra_runner)
            prev_half = (inning, top_bottom)

        home_score = result.get("homeScore", prev_home)
        away_score = result.get("awayScore", prev_away)
        outs_after = min(play.get("count", {}).get("outs", outs), 3)
        bases_after = _bases_after(play)

        pitch_data = hit_data = None
        count_data = {"balls": 0, "strikes": 0}
        play_events = play.get("playEvents", [])
        if play_events:
            pitch_data, hit_data = _pitch_hit(play_events[-1])
            count = play_events[-1].get("count", {})
            count_data = {"balls": count.get("balls", 0), "strikes": count.get("strikes", 0)}

        plays.append({
            "inning": inning,
            "top_bottom": top_bottom,
            "outs": outs,
            "runners": bases,
            "score_diff": prev_home - prev_away,
            "outs_after": outs_after,
            "runners_after": bases_after,
            "runs_on_play": (home_score - prev_home) + (away_score - prev_away),
            "home_score": home_score,
            "away_score": away_score,
            "scheduled_innings": sched,
            "extra_runner": extra_runner,
            "description": result.get("description", ""),
            "event": result.get("event", ""),
            "batter": matchup.get("batter", {}).get("fullName", ""),
            "pitcher": matchup.get("pitcher", {}).get("fullName", ""),
            "home_team": home_team,
            "away_team": away_team,
            "pitch_data": pitch_data,
            "hit_data": hit_data,
            "balls": count_data["balls"],
            "strikes": count_data["strikes"],
            **ids,
        })
        outs, bases = outs_after, bases_after
        prev_home, prev_away = home_score, away_score

    return plays


def get_game_plays(game_pk: int) -> list[dict]:
    """
    Return all completed plays from a game, suitable for Replay Mode.

    Each dict has keys:
        inning, top_bottom, outs, runners (state BEFORE the play),
        score_diff (home - away BEFORE the play),
        outs_after (0-3), runners_after, runs_on_play,
        home_score, away_score (after the play), scheduled_innings,
        extra_runner, description, event, batter, pitcher, home_team,
        away_team, pitch_data, hit_data, balls, strikes (final count)
    """
    data = _fetch(f"{_BASE}/v1.1/game/{game_pk}/feed/live")
    if data is None:
        return []
    return parse_game_plays(data)


def live_wp(state: dict) -> dict:
    """Live state + win probability analysis (analysis.full_analysis), and
    the team layer (analysis.team_layer_wp) under "team_layer" while the game
    is in progress."""
    from analysis import full_analysis, team_layer_wp

    status = state["status"]
    if status in ("Scheduled", "Pre-Game", "Warmup") or status.startswith("Delayed Start"):
        return {**state, "status_note": "Game not yet started", "win_probability": None}
    if status.startswith(("Postponed", "Cancelled", "Suspended")):
        return {**state, "status_note": status, "win_probability": None}
    if is_game_over_status(status):
        h, a = state["score_home"], state["score_away"]
        if h == a:
            return {**state, "status_note": "Game over (tie)", "win_probability": None}
        return {**state, "status_note": "Game over",
                "win_probability": 1.0 if h > a else 0.0,
                "win_probability_pct": "100.0%" if h > a else "0.0%"}

    analysis = full_analysis(
        state["inning"], state["top_bottom"], state["outs"], state["runners"],
        state["score_diff"], state["scheduled_innings"],
        state["balls"], state["strikes"], extra_runner=state["extra_runner"],
    )
    team = team_layer_wp(
        state["inning"], state["top_bottom"], state["outs"], state["runners"],
        state["score_diff"], state["scheduled_innings"], state["extra_runner"],
        state.get("home_id"), state.get("away_id"), state.get("game_date"), state.get("game_type"),
    )
    return {**state, **analysis, "team_layer": team}


def get_live_wp(game_pk: int) -> dict | None:
    """
    Fetch live game state and compute Win Probability + full analysis.

    Returns combined dict: live state + analysis.full_analysis() output.
    Returns None if game not found.
    """
    state = get_live_state(game_pk)
    if state is None:
        return None
    return live_wp(state)
