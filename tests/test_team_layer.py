"""team_layer.py and analysis.team_layer_wp: the pre-registered team-strength
layer (research/wp_layer) as the app and the API use it.

The formula must reproduce the published 2026 predictions of the frozen run
(tests/fixtures/team_layer_rows.json is a sample of them). The run differentials
must follow the training features (layer_features.team_strength): final
regular-season games, a suspended game once on its first date, games on
earlier dates only, shrunk by n / (n + 30). scripts/check_team_layer.py checks
the same against every 2025 game on the real MLB Stats API.
"""

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import analysis  # noqa: E402
import team_layer as tl  # noqa: E402


def test_formula_reproduces_published_predictions():
    rows = json.loads((ROOT / "tests/fixtures/team_layer_rows.json").read_text())
    assert len(rows) >= 40
    assert any(r["inning"] > r["sched"] for r in rows)       # extra innings
    for r in rows:
        p = tl.team_wp(r["p_engine"], r["inning"], r["half"], r["outs"], r["sched"],
                       r["d_prev"], r["d_cur"])
        assert abs(p - r["p_team"]) < 1e-12, r


def test_layer_file_is_the_frozen_run():
    lay = tl.layer()
    assert lay["K_TEAM"] == 30 and lay["check"]["max_abs_coef_diff"] == 0.0
    assert lay["engine_tables_md5"] == "ffd4b264f96c29434ba21afb4ac7c9fb"
    res = json.loads((ROOT / "research/wp_layer/published/results_2026.json").read_text())
    assert lay["coef"] == res["coef"]["TEAM"]


def test_stage_left():
    assert tl.stage_left(1, "top", 0, 9) == 1.0
    assert tl.stage_left(5, "bottom", 1, 9) == pytest.approx(1 - 28 / 54)
    assert tl.stage_left(9, "bottom", 2, 9) == pytest.approx(3 / 54)    # floored
    assert tl.stage_left(11, "top", 0, 9) == pytest.approx(3 / 54)
    assert tl.stage_left(4, "top", 0, 7) == pytest.approx(1 - 18 / 42)


def _game(pk, date, home, away, rh, ra, state="F", season=None):
    return {"gamePk": pk, "season": season or date[:4], "status": {"codedGameState": state},
            "teams": {"home": {"team": {"id": home}, "score": rh},
                      "away": {"team": {"id": away}, "score": ra}}}, date


GAMES = [
    _game(1, "2025-04-01", 1, 2, 5, 3),
    _game(2, "2025-04-02", 2, 1, 2, 4),
    _game(3, "2025-04-03", 1, 2, 9, 0, state="D"),          # postponed: not final
    _game(4, "2025-05-01", 3, 4, 1, 0),                     # suspended on 05-01 ...
    _game(4, "2025-05-02", 3, 4, 1, 0),                     # ... finished 05-02: once
    _game(5, "2025-06-01", 1, 2, 7, 0, season="2024"),      # another season's game
    _game(6, "2026-04-05", 1, 3, 6, 1),
    _game(7, "2026-04-09", 4, 2, 5, 3),
    _game(8, "2026-04-10", 1, 2, 10, 0),                    # the game's own date
]


def fake_fetch(calls, fail_first=False):
    def fetch(url):
        calls.append(url)
        if fail_first and len(calls) == 1:
            return None
        q = parse_qs(urlparse(url).query)
        lo, hi = q["startDate"][0], q["endDate"][0]
        dates = {}
        for g, d in GAMES:
            if lo <= d <= hi:
                dates.setdefault(d, []).append(g)
        return {"dates": [{"date": d, "games": gs} for d, gs in sorted(dates.items())]}
    return fetch


@pytest.fixture(autouse=True)
def clear_cache():
    tl._CACHE.clear()
    tl._FAILED.clear()
    yield
    tl._CACHE.clear()
    tl._FAILED.clear()


def test_strengths_follow_the_training_rules(monkeypatch):
    calls = []
    monkeypatch.setattr(tl, "_fetch", fake_fetch(calls))
    s = tl.strengths(1, 2, "2026-04-10")
    # last season: team 1 +2 and +2 per game, team 2 -2 and -2 (postponed and
    # other-season games left out)
    assert s["d_prev"] == pytest.approx(2 - (-2))
    # this season before 04-10: team 1 one game +5, team 2 one game -2; shrunk by n/(n+30)
    assert s["d_cur"] == pytest.approx(5 / 31 - (-2) / 31)
    assert s["teams"]["home"]["n_cur"] == 1 and s["teams"]["away"]["n_prev"] == 2
    assert any("endDate=2026-04-09" in c for c in calls)     # the day before
    # the suspended game counts once
    s34 = tl.strengths(3, 4, "2026-04-10")
    assert s34["teams"]["home"]["n_prev"] == 1


def test_first_day_of_season_has_no_current_games(monkeypatch):
    monkeypatch.setattr(tl, "_fetch", fake_fetch([]))
    s = tl.strengths(1, 2, "2026-03-01")
    assert s["d_cur"] == 0.0 and s["teams"]["home"]["n_cur"] == 0


def test_no_previous_season_or_failed_read_gives_none(monkeypatch):
    calls = []
    monkeypatch.setattr(tl, "_fetch", fake_fetch(calls, fail_first=True))
    assert tl.strengths(1, 2, "2026-04-10") is None          # the read failed
    n = len(calls)
    assert tl.strengths(1, 2, "2026-04-10") is None          # within RETRY_AFTER: no new read
    assert len(calls) == n
    monkeypatch.setattr(tl, "RETRY_AFTER", 0)
    assert tl.strengths(1, 2, "2026-04-10") is not None      # the failure was not kept
    assert tl.strengths(1, 99, "2026-04-10") is None         # team 99 has no last season
    assert tl.strengths(1, 2, "2021-06-01") is None          # before the fitted seasons


def test_current_season_is_read_again_after_an_hour(monkeypatch):
    calls = []
    monkeypatch.setattr(tl, "_fetch", fake_fetch(calls))
    assert tl._games_cached(2026, "2026-04-09", now_year=2026) is not None
    assert tl._games_cached(2025, "2025-12-31", now_year=2026) is not None
    assert tl._games_cached(2026, "2026-04-09", now_year=2026) is not None
    assert len(calls) == 2                                    # both cached
    monkeypatch.setattr(tl, "CURRENT_TTL", -1)
    tl._games_cached(2026, "2026-04-09", now_year=2026)
    tl._games_cached(2025, "2025-12-31", now_year=2026)
    assert len(calls) == 3                                    # only the current season again
    assert all("fields=" in c for c in calls)


def test_unexpected_shape_gives_none(monkeypatch):
    for bad in ({"dates": "x"}, {"dates": [{"date": "2025-04-01", "games": [{"season": "2025",
                 "status": {"codedGameState": "F"}, "gamePk": 1, "teams": None}]}]}):
        tl._CACHE.clear(); tl._FAILED.clear()
        monkeypatch.setattr(tl, "_fetch", lambda url, bad=bad: bad)
        assert tl.strengths(1, 2, "2026-04-10") is None


def test_final_game_without_score_is_not_guessed(monkeypatch):
    def fetch(url):
        g, d = _game(1, "2025-04-01", 1, 2, 5, 3)
        del g["teams"]["home"]["score"]
        return {"dates": [{"date": d, "games": [g]}]}
    monkeypatch.setattr(tl, "_fetch", fetch)
    assert tl.strengths(1, 2, "2026-04-10") is None


def test_team_layer_wp(monkeypatch):
    args = (5, "top", 1, (1, 0, 0), 1, 9, True, 1, 2, "2026-07-01")
    assert analysis.team_layer_wp(*args, "F") == {"applied": False, "reason": "not_regular_season"}
    assert analysis.team_layer_wp(*args[:7], None, 2, "2026-07-01", "R")["reason"] == "unknown_teams"
    monkeypatch.setattr(tl, "strengths", lambda h, a, d: None)
    assert analysis.team_layer_wp(*args, "R")["reason"] == "no_team_data"
    monkeypatch.setattr(tl, "strengths", lambda h, a, d: {"d_prev": 0.5, "d_cur": -0.2, "teams": {}})
    t = analysis.team_layer_wp(*args, "R")
    p = analysis.state_wp(5, "top", 1, (1, 0, 0), 1, 9, True)
    assert t["applied"] and t["engine_wp"] == p
    assert t["team_wp"] == tl.team_wp(p, 5, "top", 1, 9, 0.5, -0.2)
    assert t["team_wp"] > tl.team_wp(p, 5, "top", 1, 9, 0.0, 0.0)   # stronger home team
    # a half that has just ended is scored as the start of the next one
    e = analysis.team_layer_wp(5, "top", 3, (0, 0, 0), 1, 9, True, 1, 2, "2026-07-01", "R")
    assert e["team_wp"] == tl.team_wp(analysis.state_wp(5, "top", 3, (0, 0, 0), 1, 9, True),
                                      5, "bottom", 0, 9, 0.5, -0.2)
    e = analysis.team_layer_wp(5, "bottom", 3, (0, 0, 0), 1, 9, True, 1, 2, "2026-07-01", "R")
    assert e["team_wp"] == tl.team_wp(analysis.state_wp(5, "bottom", 3, (0, 0, 0), 1, 9, True),
                                      6, "top", 0, 9, 0.5, -0.2)
    # the engine input is clipped to +-15 runs, as in the fit
    big = analysis.team_layer_wp(2, "top", 0, (0, 0, 0), -18, 9, True, 1, 2, "2026-07-01", "R")
    assert big["engine_wp"] == analysis.state_wp(2, "top", 0, (0, 0, 0), -15, 9, True)
    assert analysis.team_layer_wp(*args[:9], "2021-07-01", "R")["reason"] == "before_fitted_seasons"
    # a failure inside the lookup becomes "no data", never an exception
    monkeypatch.setattr(tl, "strengths", lambda h, a, d: {}["missing"])
    assert analysis.team_layer_wp(*args, "R")["reason"] == "no_team_data"
