"""The live card shows the bench options (bench_live) for a game in progress (offline AppTest)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import sys
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer
from test_bench_live import feed

LIVE = {"inning": 7, "top_bottom": "bottom", "outs": 1, "runners": (1, 0, 0), "score_home": 3,
        "score_away": 3, "score_diff": 0, "status": "In Progress", "away_team": "A", "home_team": "H",
        "batter_name": "b", "pitcher_name": "p", "balls": 0, "strikes": 0, "scheduled_innings": 9,
        "extra_runner": True, "home_id": 121, "away_id": 143, "game_date": "2027-04-01", "game_type": "R"}
live_feed.get_todays_games = lambda date=None: [{"gamePk": 7, "away_team": "A", "home_team": "H",
                                                 "status": "In Progress", "start_time_utc": ""}]
live_feed.get_live_state = lambda pk: dict(LIVE)
team_layer.strengths = lambda h, a, d: {"d_prev": 0.0, "d_cur": 0.0, "teams": {}}
calls = []
def fake_feed(pk):
    calls.append(pk)
    return feed()
bench_live.get_feed = fake_feed

lang = sys.argv[1]
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
if lang == "JA":
    at.sidebar.radio[0].set_value("JA").run()
at.button(key="fetch_games").click().run()
at.button(key="load_live").click().run()
assert not at.exception, at.exception
assert calls == [7], calls
frames = [d.value for d in at.dataframe]
assert len(frames) == 2, len(frames)
off, dfn = frames
col = off.columns[0]
assert ("送りバント" if lang == "JA" else "bunt") in list(off[col]), list(off[col])
assert any(str(x).startswith("代打" if lang == "JA" else "pinch_hit") for x in off[col])
assert ("敬遠" if lang == "JA" else "intentional_walk") in list(dfn[dfn.columns[0]])
cap = " ".join(c.value for c in at.caption)
assert "2026-09-27" in cap, cap
# a rerun without a new fetch does not recompute (no second feed call)
at.run()
assert calls == [7], calls
# a game that is over shows no bench panel
LIVE["status"] = "Final"
at.button(key="load_live").click().run()
assert not at.exception, at.exception
assert len(at.dataframe) == 0
print("OK", lang)
"""


@pytest.mark.parametrize("lang", ["EN", "JA"])
def test_bench_panel(lang):
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code, lang], env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert f"OK {lang}" in r.stdout
