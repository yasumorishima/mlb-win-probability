"""Look-back view: finding a game by team, and the key-moment buttons."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_key_moments_are_the_biggest_swings_for_the_batting_team_in_game_order():
    import bench_review as BR
    tl = [{"top_bottom": "top"}, {"top_bottom": "bottom"}, {"top_bottom": "top"}, {"top_bottom": "bottom"}]
    wp = [0.50, 0.45, 0.80, 0.78, 0.30]          # home WP before each PA, then the end
    series = [{"wp": w} for w in wp]
    km = BR.key_moments(tl, series, n=2)
    assert [k for k, _ in km] == [1, 3]          # +35 for home (bottom), -48 for home in the bottom
    assert round(km[0][1]) == 35 and round(km[1][1]) == -48
    km1 = BR.key_moments(tl, series, n=3)
    assert [k for k, _ in km1] == [0, 1, 3] and round(km1[0][1]) == 5   # top half: away's view


def test_team_games_parse_and_order(monkeypatch):
    import live_feed as L
    data = {"dates": [{"date": "2026-09-01", "games": [
        {"gamePk": 1, "officialDate": "2026-09-01", "gameType": "R", "status": {"detailedState": "Final"},
         "teams": {"home": {"team": {"id": 110, "name": "Baltimore Orioles"}, "score": 3},
                   "away": {"team": {"id": 147, "name": "New York Yankees"}, "score": 5}}}]},
        {"date": "2026-09-02", "games": [
        {"gamePk": 2, "officialDate": "2026-09-02", "gameType": "R", "status": {"detailedState": "Final"},
         "teams": {"home": {"team": {"id": 147, "name": "New York Yankees"}, "score": 1},
                   "away": {"team": {"id": 110, "name": "Baltimore Orioles"}, "score": 4}}}]}]}
    monkeypatch.setattr(L, "_fetch", lambda url: data)
    g = L.get_team_games(110, "2026-09-01", "2026-09-30")
    assert [x["gamePk"] for x in g] == [2, 1]
    import bench_review as BR
    x = BR.TXT["JA"]
    abbr = {110: "BAL", 147: "NYY"}
    assert BR._team_label(g[0], 110, abbr, x) == "09/02 @ NYY  4-1 勝"
    assert BR._team_label(g[1], 110, abbr, x) == "09/01 vs NYY  3-5 負"


CHILD = r"""
import sys
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer
import game_fixture as F
FEED = F.game_feed()
live_feed.get_teams = lambda: [{"id": 1, "abbr": "HOM", "name": "Home"}, {"id": 2, "abbr": "AWY", "name": "Away"}]
live_feed.get_team_games = lambda tid, a, b: [{"gamePk": 777, "date": "2026-07-01", "home": "Home", "away": "Away",
    "home_id": 1, "away_id": 2, "home_score": 1, "away_score": 0, "status": "Final", "game_type": "R"}]
team_layer.strengths = lambda h, a, d: {"d_prev": 0.1, "d_cur": 0.0, "teams": {}}
bench_live.get_feed = lambda pk: FEED
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("JA").run()
at.sidebar.radio[1].set_value("📋 Bench").run()
at.radio(key="bench_view_JA").set_value("振り返り").run()
at.radio(key="review_find_JA").set_value("⚾ チームから").run()
assert not at.exception, at.exception
sel = [s for s in at.selectbox if s.key and s.key.startswith("review_tgame_")][0]
assert "07/01 vs AWY  1-0 勝" in sel.options[0], sel.options
md = " ".join(m.value for m in at.markdown)
assert "実際には" in md
km = [b for b in at.button if b.key and b.key.startswith("review_km_777_")]
assert 1 <= len(km) <= 5, [b.key for b in at.button]
target = int(km[-1].key.rsplit("_", 1)[1])
km[-1].click().run()
assert not at.exception, at.exception
assert at.session_state["review_k"] == target, (at.session_state["review_k"], target)
print("OK")
"""


def test_find_by_team_and_jump_to_a_key_moment():
    pytest.importorskip("streamlit")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "OK" in r.stdout
