"""
Team-strength layer on top of the engine (research/wp_layer, pre-registered
and confirmed on 2026: research/wp_layer/RESULTS.md).

    p_team = sigmoid(intercept + sum(coef[t] * term[t]))

with the terms of research/wp_layer/published/layer_team.json, built from the
engine's home WP, the stage of the game and the two teams' run differentials
per game (d_prev: previous regular season; d_cur: this regular season on
earlier dates, shrunk by n / (n + K_TEAM)). The layer was fitted and tested
on regular-season games only, so it is only applied to them.

The run differentials come from the MLB Stats API schedule, the same source
and the same rules as the training tables (scripts/build_states.py): final
regular-season games, a suspended game counted once on its first date.
Stdlib only, like live_feed.py.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import date as _date, timedelta
from functools import lru_cache
from pathlib import Path

from live_feed import _BASE, _fetch

LAYER_PATH = Path(__file__).resolve().parent / "research/wp_layer/published/layer_team.json"
# The layer was fitted on 2022-2025 (2021 only as 2022's previous season) and
# tested on 2026; it is not applied to earlier seasons.
FIRST_SEASON = 2022
# Only the fields the run differentials need (3.2 MB -> about 0.4 MB a season).
_FIELDS = "dates,date,games,gamePk,season,status,codedGameState,teams,home,away,team,id,score"


@lru_cache(maxsize=1)
def layer() -> dict:
    return json.loads(LAYER_PATH.read_text())


def stage_left(inning: int, half: str, outs: int, sched: int = 9) -> float:
    """r: share of regulation outs still to be played, floored at one half-inning."""
    total = 6 * sched
    done = ((inning - 1) * 2 + (half == "bottom")) * 3 + outs
    return max(1 - done / total, 3 / total)


def team_wp(p_engine: float, inning: int, half: str, outs: int, sched: int,
            d_prev: float, d_cur: float) -> float:
    """Home WP of the TEAM layer."""
    lay = layer()
    clip = lay["logit_clip"]
    p = min(max(p_engine, clip), 1 - clip)
    L = math.log(p / (1 - p))
    r = stage_left(inning, half, outs, sched)
    term = {"L": L, "Lr": L * r, "Lr2": L * r * r, "r": r, "r2": r * r,
            "dprev_r": d_prev * r, "dprev_r2": d_prev * r * r,
            "dcur_r": d_cur * r, "dcur_r2": d_cur * r * r}
    z = lay["intercept"] + sum(lay["coef"][t] * term[t] for t in lay["terms"])
    return 1 / (1 + math.exp(-z))


def season_games(season: int, end: str) -> list[dict] | None:
    """Final regular-season games of a season up to and including `end`
    (YYYY-MM-DD), one row per game on its first date. None if the API fails."""
    data = _fetch(f"{_BASE}/v1/schedule?sportId=1&gameType=R"
                  f"&startDate={season}-03-01&endDate={end}&fields={_FIELDS}")
    if data is None:
        return None
    seen, out = set(), []
    try:
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if str(g.get("season")) != str(season):
                    continue
                if g.get("status", {}).get("codedGameState") != "F":
                    continue
                pk = g.get("gamePk")
                if pk in seen:
                    continue
                seen.add(pk)
                h, a = g["teams"]["home"], g["teams"]["away"]
                out.append({"game_pk": pk, "date": d["date"],
                            "home": int(h["team"]["id"]), "away": int(a["team"]["id"]),
                            "rs_home": int(h["score"]), "rs_away": int(a["score"])})
    except (AttributeError, KeyError, TypeError, ValueError):
        return None   # an unexpected shape, or a final game without its score: do not guess
    return out


def _per_game(games: list[dict], team: int) -> tuple[float, int]:
    s = n = 0
    for g in games:
        if g["home"] == team:
            s, n = s + g["rs_home"] - g["rs_away"], n + 1
        elif g["away"] == team:
            s, n = s + g["rs_away"] - g["rs_home"], n + 1
    return s, n


_CACHE: dict = {}          # (season, end) -> (time read, games)
_FAILED: dict = {}         # (season, end) -> time of the last failed read
_CACHE_MAX = 16
_LOCK = threading.Lock()
CURRENT_TTL = 3600         # a season still being played is read again after an hour
RETRY_AFTER = 60           # after a failed read, wait this long before trying again


def _games_cached(season: int, end: str, now_year: int | None = None) -> list[dict] | None:
    """season_games, cached per (season, end). A finished season is kept; the
    current season is read again after CURRENT_TTL (a suspended game can turn
    final during the day). A failed read is not kept, but the next try waits
    RETRY_AFTER so a live page does not block on every refresh during an outage."""
    key = (season, end)
    now = time.time()
    current = season >= (now_year or _date.today().year)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and not (current and now - hit[0] > CURRENT_TTL):
            return hit[1]
        if now - _FAILED.get(key, -1e18) < RETRY_AFTER:
            return None
    g = season_games(season, end)
    with _LOCK:
        if g is None:
            _FAILED[key] = now
            return None
        _FAILED.pop(key, None)
        _CACHE.pop(key, None)
        while len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (now, g)
    return g


def strengths(home_id: int, away_id: int, game_date: str) -> dict | None:
    """d_prev and d_cur (home minus away) for a game on game_date, from games
    on earlier dates only. None if the schedule cannot be read or a team has
    no previous regular season."""
    day = _date.fromisoformat(game_date)
    season = day.year
    if season < FIRST_SEASON:
        return None
    prev_end = f"{season - 1}-12-31"
    cur_end = (day - timedelta(days=1)).isoformat()
    prev = _games_cached(season - 1, prev_end)
    cur = _games_cached(season, cur_end) if cur_end >= f"{season}-03-01" else []
    if prev is None or cur is None:
        return None
    cur = [g for g in cur if g["date"] < game_date]
    k = layer()["K_TEAM"]
    out = {}
    for side, team in (("home", home_id), ("away", away_id)):
        s, n = _per_game(prev, team)
        if n == 0:
            return None
        cs, cn = _per_game(cur, team)
        out[side] = {"prev": s / n, "cur": (cs / cn) * cn / (cn + k) if cn else 0.0,
                     "n_prev": n, "n_cur": cn}
    return {"d_prev": out["home"]["prev"] - out["away"]["prev"],
            "d_cur": out["home"]["cur"] - out["away"]["cur"], "teams": out}
