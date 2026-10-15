"""
What the app and the API show, computed from the fitted engine.

`wp_engine.Engine` (tables in `engine_tables.json`, fitted by
scripts/fit_engine.py on real plate-appearance transitions) holds the
numbers. This module looks them up and shapes them for the Streamlit app
(streamlit_app.py), the FastAPI service (api.py) and the live feed
(live_feed.py):

* home win probability, and the batting team's win probability
* Leverage Index (expected |WP change| of the next plate appearance over
  the league average) with a label
* run expectancy and P(score at least once) for the 24 base-out states,
  from the engine's run distributions for the rest of the half
* win probability after each kind of plate-appearance outcome, averaged
  over what actually happened on such outcomes in that base-out state
* tactics: stolen-base break-even rates; sacrifice bunt and intentional
  walk as a change in win probability, each from one side's point of view
* preset scenarios, and the inputs for the Claude bench coach
  (strategist.advise)

Every player is league-average to the engine; nothing here adjusts for
who is batting or pitching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from wp_engine import END, MAX_RUNS, Engine, _state

TABLES_PATH = Path(__file__).resolve().parent / "engine_tables.json"
HALVES = ("top", "bottom")
RUNNER_LABELS = {(0, 0, 0): "---", (1, 0, 0): "1--", (0, 1, 0): "-2-", (1, 1, 0): "12-",
                 (0, 0, 1): "--3", (1, 0, 1): "1-3", (0, 1, 1): "-23", (1, 1, 1): "123"}

TACTIC_NAMES = {
    "steal_2b": ("Steal 2nd base", "二盗"),
    "steal_3b": ("Steal 3rd base", "三盗"),
    "sacrifice_bunt": ("Sacrifice bunt", "送りバント"),
    "intentional_walk": ("Intentional walk", "敬遠"),
}

# Plate-appearance outcome classes of the engine (wp_engine.CLASSES) shown in
# the what-if panel. "OUT" is every ball-in-play out, double plays included.
OUTCOMES = [
    ("K", "Strikeout", "三振"),
    ("OUT", "Out in play", "凡打（併殺含む）"),
    ("BB", "Walk / HBP", "四死球"),
    ("1B", "Single", "単打"),
    ("2B", "Double", "二塁打"),
    ("3B", "Triple", "三塁打"),
    ("HR", "Home run", "本塁打"),
]


@lru_cache(maxsize=2)
def get_engine(extra_runner: bool = True) -> Engine:
    """The engine; `extra_runner=False` for games without the automatic
    runner on second in extra innings (postseason)."""
    return Engine.load(TABLES_PATH, extra_runner=extra_runner)


def engine_info() -> dict:
    t = get_engine().t
    return {"engine": "wp_engine", "tables": TABLES_PATH.name,
            "train_seasons": t.get("train_seasons"), "n_pa": t.get("n_pa")}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _check(inning, half, outs, runners, max_outs=2):
    if half not in HALVES:
        raise ValueError(f"half must be 'top' or 'bottom', got {half!r}")
    if isinstance(inning, bool) or not isinstance(inning, (int, np.integer)) or inning < 1:
        raise ValueError(f"inning must be a positive integer, got {inning!r}")
    if outs not in range(max_outs + 1):
        raise ValueError(f"outs must be 0..{max_outs}, got {outs!r}")
    r = tuple(int(x) for x in runners)
    if len(r) != 3 or any(x not in (0, 1) for x in r):
        raise ValueError(f"runners must be three 0/1 flags, got {runners!r}")
    return r


def _count(balls, strikes):
    if balls is None or strikes is None:
        return None, None
    if balls not in range(4) or strikes not in range(3):
        raise ValueError(f"count must be balls 0..3 and strikes 0..2, got {balls}-{strikes}")
    return int(balls), int(strikes)


def game_over(inning: int, half: str, score_diff: int, sched: int = 9) -> bool:
    """Home team leads in the bottom of the last scheduled inning or later:
    the game is over (or the bottom half is not played)."""
    return half == "bottom" and inning >= sched and score_diff > 0


# ---------------------------------------------------------------------------
# Win probability and leverage
# ---------------------------------------------------------------------------

def home_wp(inning, half, outs, runners, score_diff, sched=9, balls=None, strikes=None,
            extra_runner=True) -> float:
    """Home win probability before the next pitch (count-aware when given)."""
    r = _check(inning, half, outs, runners)
    balls, strikes = _count(balls, strikes)
    eng = get_engine(extra_runner)
    if balls is None:
        return eng.wp(inning, half, outs, r, score_diff, sched)
    return eng.wp_at_count(inning, half, outs, r, score_diff, balls, strikes, sched)


def batting_wp(home: float, half: str) -> float:
    """Win probability of the team at bat (visitors in the top half)."""
    return home if half == "bottom" else 1.0 - home


def state_wp(inning, half, outs, runners, score_diff, sched=9, extra_runner=True) -> float:
    """Home WP for a base-out state; outs == 3 means the half has just ended."""
    r = _check(inning, half, outs, runners, max_outs=3)
    eng = get_engine(extra_runner)
    if outs == 3:
        return float(eng._wp_after(inning, half, END, 0, score_diff, sched))
    return eng.wp(inning, half, outs, r, score_diff, sched)


def leverage_index(inning, half, outs, runners, score_diff, sched=9, extra_runner=True) -> float:
    r = _check(inning, half, outs, runners)
    eng = get_engine(extra_runner)
    return float(eng.leverage(inning, half, outs, r, score_diff, sched,
                              league_swing=eng.t["league_swing"]))


def li_label(li: float) -> str:
    """Human-readable label for the Leverage Index."""
    if li < 0.5:
        return "Low"
    if li < 1.5:
        return "Medium"
    if li < 3.0:
        return "High"
    return "Very High"


@dataclass
class GameState:
    inning: int
    top_bottom: str          # "top" or "bottom"
    outs: int                # 0-2 (3 = the half has just ended)
    runners: tuple[int, int, int]
    score_diff: int          # home - away
    scheduled_innings: int = 9


def wpa(before: GameState, after: GameState, extra_runner=True) -> dict:
    """Change in home win probability between two states."""
    w0 = state_wp(before.inning, before.top_bottom, before.outs, before.runners,
                  before.score_diff, before.scheduled_innings, extra_runner)
    w1 = state_wp(after.inning, after.top_bottom, after.outs, after.runners,
                  after.score_diff, after.scheduled_innings, extra_runner)
    return {"wpa": round(w1 - w0, 4), "wp_before": round(w0, 4), "wp_after": round(w1, 4)}


def wp_after_play(inning, half, outs_after, runners_after, runs, score_diff_before, sched=9,
                  extra_runner=True) -> float:
    """Home WP right after a play that left `outs_after` outs and `runners_after`
    on base with `runs` scored (score_diff_before = home - away before it)."""
    r = _check(inning, half, min(outs_after, 3), runners_after, max_outs=3)
    nxt = END if outs_after >= 3 else _state(outs_after, *r)
    return float(get_engine(extra_runner)._wp_after(inning, half, nxt, int(runs),
                                                     score_diff_before, sched))


# ---------------------------------------------------------------------------
# Run expectancy (24 base-out states)
# ---------------------------------------------------------------------------

def run_expectancy(half, outs, runners) -> tuple[float, float]:
    """(expected runs in the rest of the half, P(at least one run))."""
    r = _check(1, half, outs, runners)
    dist = get_engine().runs[half][_state(outs, *r)]
    return float(np.arange(MAX_RUNS + 1) @ dist), float(1.0 - dist[0])


def re24_table(half: str = "top") -> list[dict]:
    """Run expectancy for the 24 base-out states for one half (visitors
    batting in the top, home in the bottom)."""
    rows = []
    for runners, label in RUNNER_LABELS.items():
        for outs in range(3):
            er, p = run_expectancy(half, outs, runners)
            rows.append({"runners": label, "runner1": runners[0], "runner2": runners[1],
                         "runner3": runners[2], "outs": outs, "expected_runs": round(er, 3),
                         "p_score": round(p, 3)})
    return rows


# ---------------------------------------------------------------------------
# What-if and tactics
# ---------------------------------------------------------------------------

def what_if(inning, half, outs, runners, score_diff, sched=9, wp_now=None,
            extra_runner=True) -> list[dict]:
    """WP after the plate appearance ends in each outcome class.

    Each value is the average over the observed transitions for that class
    in this base-out state (runner advances, double plays and runners thrown
    out included). `delta` is the change for the batting team from `wp_now`
    (home WP; defaults to the count-free WP)."""
    r = _check(inning, half, outs, runners)
    eng = get_engine(extra_runner)
    if wp_now is None:
        wp_now = eng.wp(inning, half, outs, r, score_diff, sched)
    s = _state(outs, *r)
    out = []
    for cls, name, name_ja in OUTCOMES:
        rows = eng.by_class.get((half, s, cls))
        if not rows:
            continue
        w = float(sum(p * eng._wp_after(inning, half, nxt, runs, score_diff, sched)
                      for nxt, runs, p in rows))
        out.append({"outcome": cls, "name": name, "name_ja": name_ja,
                    "home_wp": round(w, 4),
                    "batting_wp": round(batting_wp(w, half), 4),
                    "delta": round(batting_wp(w, half) - batting_wp(wp_now, half), 4)})
    return out


def tactics(inning, half, outs, runners, score_diff, sched=9, balls=None, strikes=None,
            extra_runner=True) -> list[dict]:
    """Engine tactics. Steals report the success rate needed to break even;
    bunts and intentional walks report the change in win probability.
    `side` says whose win probability the numbers are (offense = batting team)."""
    r = _check(inning, half, outs, runners)
    balls, strikes = _count(balls, strikes)
    if game_over(inning, half, score_diff, sched):
        return []
    batting = "home" if half == "bottom" else "away"
    fielding = "away" if batting == "home" else "home"
    out = []
    for t in get_engine(extra_runner).tactics(inning, half, outs, r, score_diff, sched,
                                              balls=balls, strikes=strikes):
        name, name_ja = TACTIC_NAMES.get(t.tactic, (t.tactic, t.tactic))
        note = t.note
        if t.tactic == "sacrifice_bunt" and balls is not None and (balls, strikes) != (0, 0):
            note += ("; the bunt outcomes are pooled over all counts but compared with "
                     "the WP at this count")
        out.append({
            "tactic": t.tactic, "name": name, "name_ja": name_ja,
            "side": t.side, "team": batting if t.side == "offense" else fielding,
            "wp_now": round(t.wp_now, 4),
            "wp_if": None if t.wp_if is None else round(t.wp_if, 4),
            "delta": None if t.delta is None else round(t.delta, 4),
            "breakeven": None if t.breakeven is None else round(t.breakeven, 4),
            "n": t.n, "note": note,
        })
    return out


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def full_analysis(inning, top_bottom, outs, runners, score_diff, scheduled_innings=9,
                  balls=None, strikes=None, extra_runner=True) -> dict:
    r = _check(inning, top_bottom, outs, runners)
    balls, strikes = _count(balls, strikes)
    sched = scheduled_innings
    over = game_over(inning, top_bottom, score_diff, sched)
    wp = home_wp(inning, top_bottom, outs, r, score_diff, sched, balls, strikes, extra_runner)
    li = 0.0 if over else leverage_index(inning, top_bottom, outs, r, score_diff, sched,
                                         extra_runner)
    er, p_score = run_expectancy(top_bottom, outs, r)
    return {
        "game_state": {
            "inning": inning, "top_bottom": top_bottom, "outs": outs,
            "runners": {"1B": r[0], "2B": r[1], "3B": r[2]},
            "score_diff": score_diff, "scheduled_innings": sched,
            "balls": balls, "strikes": strikes,
        },
        "game_over": over,
        "win_probability": round(wp, 4),
        "win_probability_pct": f"{wp * 100:.1f}%",
        "batting_team": "home" if top_bottom == "bottom" else "away",
        "batting_team_wp": round(batting_wp(wp, top_bottom), 4),
        "leverage_index": round(li, 2),
        "leverage_label": li_label(li),
        "run_expectancy": round(er, 3),
        "p_score": round(p_score, 3),
        "tactics": tactics(inning, top_bottom, outs, r, score_diff, sched, balls, strikes,
                           extra_runner),
        "what_if": [] if over else what_if(inning, top_bottom, outs, r, score_diff, sched,
                                           wp_now=wp, extra_runner=extra_runner),
        "model": engine_info(),
    }


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "ninth_inning_drama": {
        "name": "9th Inning Drama", "name_ja": "9回裏2アウト満塁同点",
        "description": "Bottom 9th, 2 outs, bases loaded, tie game",
        "state": GameState(9, "bottom", 2, (1, 1, 1), 0),
    },
    "game_start": {
        "name": "Game Start", "name_ja": "1回表開始",
        "description": "Top of 1st, no outs, bases empty, 0-0",
        "state": GameState(1, "top", 0, (0, 0, 0), 0),
    },
    "rally_7th": {
        "name": "7th Inning Rally", "name_ja": "7回裏1点ビハインド1死1-2塁",
        "description": "Bottom 7th, 1 out, runners on 1st & 2nd, down by 1",
        "state": GameState(7, "bottom", 1, (1, 1, 0), -1),
    },
    "tied_8th": {
        "name": "Tied 8th", "name_ja": "8回表同点無死走者なし",
        "description": "Top of 8th, no outs, bases empty, tie game",
        "state": GameState(8, "top", 0, (0, 0, 0), 0),
    },
    "walkoff_chance": {
        "name": "Walk-off Chance", "name_ja": "9回裏サヨナラ1死2-3塁",
        "description": "Bottom 9th, 1 out, runners on 2nd & 3rd, tie game",
        "state": GameState(9, "bottom", 1, (0, 1, 1), 0),
    },
    "comfortable_lead": {
        "name": "Comfortable Lead", "name_ja": "5回表3点リード",
        "description": "Top of 5th, no outs, bases empty, home team up by 3",
        "state": GameState(5, "top", 0, (0, 0, 0), 3),
    },
}


def analyze_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}")
    sc = SCENARIOS[name]
    s = sc["state"]
    return {"scenario": sc["name"], "scenario_ja": sc["name_ja"],
            "description": sc["description"],
            **full_analysis(s.inning, s.top_bottom, s.outs, s.runners, s.score_diff,
                            s.scheduled_innings)}


# ---------------------------------------------------------------------------
# Claude bench coach (strategist.advise)
# ---------------------------------------------------------------------------

def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def advice_inputs(analysis: dict, lang: str = "EN", players: dict | None = None):
    """(situation, engine, players) for strategist.advise from a full_analysis()
    result. Numbers are rounded to 3 decimals so cited values can be checked."""
    gs = analysis["game_state"]
    situation = {
        "inning": gs["inning"], "half": gs["top_bottom"], "outs": gs["outs"],
        "runner_on_first": gs["runners"]["1B"], "runner_on_second": gs["runners"]["2B"],
        "runner_on_third": gs["runners"]["3B"],
        "score_home_minus_away": gs["score_diff"],
        "batting_team": analysis["batting_team"],
        "scheduled_innings": gs["scheduled_innings"],
    }
    if gs["balls"] is not None:
        situation["balls"], situation["strikes"] = gs["balls"], gs["strikes"]
    if lang == "JA":
        situation["reply_language"] = "Japanese"
    engine = {
        "assumption": "every player is league-average",
        "home_wp_now": round(analysis["win_probability"], 3),
        "batting_team_wp_now": round(analysis["batting_team_wp"], 3),
        "leverage_index": round(analysis["leverage_index"], 2),
        "expected_runs_rest_of_half": round(analysis["run_expectancy"], 3),
        "p_score_rest_of_half": round(analysis["p_score"], 3),
    }
    for t in analysis["tactics"]:
        d = {"side": t["side"], "wp_now": round(t["wp_now"], 3)}
        if t["breakeven"] is not None:
            d["breakeven_success_rate"] = round(t["breakeven"], 3)
        if t["wp_if"] is not None:
            d["wp_if"] = round(t["wp_if"], 3)
            d["delta_wp"] = round(t["delta"], 3)
        if t["n"] is not None:
            d["n_observed"] = t["n"]
        d["note"] = t["note"]
        engine[t["tactic"]] = d
    if players is None:
        players = {"note": "no player facts are available"}
    return situation, engine, players


def get_advice(analysis: dict, lang: str = "EN", players: dict | None = None):
    """(Advice | None, citation problems, payload). Calls the API only when
    ANTHROPIC_API_KEY is set; otherwise returns None without importing anthropic."""
    situation, engine, players = advice_inputs(analysis, lang, players)
    payload = {"situation": situation, "engine": engine, "players": players}
    if not has_api_key():
        return None, ["ANTHROPIC_API_KEY is not set"], payload
    import strategist
    adv, problems = strategist.advise(situation, engine, players)
    return adv, problems, payload
