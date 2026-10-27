"""Bench screen (bench_screen.py): one tap after the mode, a game in progress is followed (offline AppTest)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import sys, time
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer
from test_bench_live import feed

GAMES = [{"gamePk": 5, "away_team": "Away", "home_team": "Home", "status": "Final", "start_time_utc": ""},
         {"gamePk": 7, "away_team": "Away", "home_team": "Home", "status": "In Progress", "start_time_utc": ""}]
live_feed.get_todays_games = lambda date=None: list(GAMES)
team_layer.strengths = lambda h, a, d: {"d_prev": 0.2, "d_cur": 0.1, "teams": {}}
calls = []
def fake_feed(pk):
    calls.append(pk)
    f = feed()
    f["gameData"]["datetime"] = {"officialDate": "2027-04-01"}
    if pk == 5:
        f["gameData"]["status"]["detailedState"] = "Final"
    return f
bench_live.get_feed = fake_feed

lang = sys.argv[1]
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
if lang == "JA":
    at.sidebar.radio[0].set_value("JA").run()
t0 = time.time()
at.sidebar.radio[1].set_value("📋 Bench").run()            # the only tap
dt = time.time() - t0
assert not at.exception, at.exception
assert calls == [7], calls                                   # the game in progress is chosen by itself
assert at.selectbox(key="bench_select").value == 7
frames = [d.value for d in at.dataframe]
assert len(frames) == 3, len(frames)                         # matchups, offense, defense
mu = frames[0]
assert len(mu) == 4                                          # now + next 3
md = " ".join(m.value for m in at.markdown)
assert "%" in md and ("重要度" if lang == "JA" else "Leverage") in md
assert ("選手込み" if lang == "JA" else "with players") in md
print("RENDER_SECONDS", round(dt, 2))
# a tick 30 s later fetches again
at.session_state["bench_last_try"] = 0.0
at.run()
assert not at.exception, at.exception
assert calls == [7, 7], calls
# picking a finished game says so instead of showing options
at.selectbox(key="bench_select").set_value(5).run()
assert not at.exception, at.exception
assert calls[-1] == 5, calls
assert any("Final" in i.value for i in at.info), [i.value for i in at.info]
assert len(at.dataframe) == 0
print("OK", lang)
"""


@pytest.mark.parametrize("lang", ["EN", "JA"])
def test_bench_screen(lang):
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code, lang], env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert f"OK {lang}" in r.stdout
    print([l for l in r.stdout.splitlines() if l.startswith("RENDER")])
