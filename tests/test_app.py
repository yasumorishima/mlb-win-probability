"""Streamlit smoke test: the main page renders in both languages without exceptions.

Each scenario runs in its own process (one app per process), so state left
by one AppTest cannot hide or cause a failure in another. Nothing here
calls the network or the Claude API.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import sys
sys.path.insert(0, ROOT)
from streamlit.testing.v1 import AppTest

lang, with_key = sys.argv[1], sys.argv[2] == "key"
if with_key:
    import strategist
    from strategist import Advice, Citation

    calls = []

    def fake_advise(situation, engine, players):
        calls.append(1)
        adv = Advice(recommendation="intentional_walk", side="defense", confidence=0.55,
                     follows_engine=True, reason="<b>Walk him</b>; first base is open.",
                     departure_reason="",
                     cited=[Citation(name="leverage_index", value=engine["leverage_index"])])
        return adv, ["made-up problem for the test"]

    strategist.advise = fake_advise

import live_feed
fetches = []
LIVE = {"inning": 5, "top_bottom": "top", "outs": 1, "runners": (1, 0, 0), "score_home": 2,
        "score_away": 1, "score_diff": 1, "status": "In Progress", "away_team": "A",
        "home_team": "H", "batter_name": "b", "pitcher_name": "p", "balls": 1, "strikes": 1,
        "scheduled_innings": 9, "extra_runner": True}

def fake_games(date=None):
    return [{"gamePk": 1, "away_team": "A", "home_team": "H", "status": "In Progress",
             "start_time_utc": ""}]

def fake_state(pk):
    fetches.append(pk)
    return dict(LIVE)

live_feed.get_todays_games = fake_games
live_feed.get_live_state = fake_state

# the team layer, without the network: fixed run differentials
import team_layer
team_layer.strengths = lambda h, a, d: {"d_prev": 0.5, "d_cur": 0.25, "teams": {}}
LIVE.update({"home_id": 1, "away_id": 2, "game_date": "2026-07-01", "game_type": "R"})
TEAM_WORD = "Team-adjusted" if lang == "EN" else "チーム力込み"
PLAY = {"inning": 6, "top_bottom": "bottom", "outs": 0, "runners": (0, 0, 0), "score_diff": -1,
        "outs_after": 1, "runners_after": (0, 0, 0), "runs_on_play": 0, "home_score": 1,
        "away_score": 2, "scheduled_innings": 9, "extra_runner": True, "description": "K",
        "event": "Strikeout", "batter": "b", "pitcher": "p", "home_team": "H", "away_team": "A",
        "pitch_data": None, "hit_data": None, "balls": 0, "strikes": 0,
        "home_id": 1, "away_id": 2, "game_date": "2026-07-01", "game_type": "R"}
live_feed.get_game_plays = lambda pk: [dict(PLAY)]

import streamlit
fragments = []
_real_fragment = streamlit.fragment
def spy_fragment(func=None, *, run_every=None):
    fragments.append(run_every)
    return _real_fragment(func, run_every=run_every)
streamlit.fragment = spy_fragment

at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=120)
at.run()
assert not at.exception, at.exception
if lang == "JA":
    at.sidebar.radio[0].set_value("JA").run()
    assert not at.exception, at.exception
title = "MLB 勝利確率エンジン" if lang == "JA" else "MLB Win Probability Engine"
md = " ".join(m.value for m in at.markdown)
assert title in md, md[:300]
assert "<svg" in md
infos = " ".join(i.value for i in at.info)
if with_key:
    assert "ANTHROPIC_API_KEY" not in infos
else:
    assert "ANTHROPIC_API_KEY" in infos, infos
    assert not any(b.key == "ask_coach" for b in at.button)

# a preset sets the widgets, and the analysis follows them
at.button(key="preset_rally_7th").click().run()
assert not at.exception, at.exception
assert at.selectbox(key="w_inning").value == 7
assert at.selectbox(key="w_half_" + lang).value == "bottom"
assert at.selectbox(key="w_half_" + lang).options == (["表", "裏"] if lang == "JA" else ["Top", "Bottom"])
assert at.checkbox(key="w_r1").value and at.checkbox(key="w_r2").value
assert at.slider(key="w_diff").value == -1
md = " ".join(m.value for m in at.markdown)
assert ("送りバント" if lang == "JA" else "Sacrifice bunt") in md
# the tactic notes follow the language
assert ("ファウルになったバント" if lang == "JA" else "bunts fouled off") in md
assert ("bunts fouled off" if lang == "JA" else "ファウルになったバント") not in md

# the preset values survive another interaction (widgets keep them)
at.selectbox(key="w_balls").set_value(3).run()
assert not at.exception, at.exception
assert at.selectbox(key="w_inning").value == 7

# the half survives a language switch
other = "EN" if lang == "JA" else "JA"
at.sidebar.radio[0].set_value(other).run()
assert not at.exception, at.exception
assert at.selectbox(key="w_half_" + other).value == "bottom"
at.sidebar.radio[0].set_value(lang).run()
assert at.selectbox(key="w_half_" + lang).value == "bottom"

if with_key:
    at.button(key="ask_coach").click().run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert ("敬遠" if lang == "JA" else "Intentional walk") in md
    assert "&lt;b&gt;Walk him&lt;/b&gt;" in md          # model text is escaped
    assert any("made-up problem" in w.value for w in at.warning)
    # the per-session cap stops further calls
    n = len(calls)
    at.session_state["coach_calls"] = 10
    at.button(key="ask_coach").click().run()
    assert not at.exception, at.exception
    assert len(calls) == n
    assert any("10" in w.value and ("上限" if lang == "JA" else "limit") in w.value
               for w in at.warning), [w.value for w in at.warning]

# a steal no rate can pay for: the card says so once, and the note keeps its other parts
at.selectbox(key="w_inning").set_value(9).run()
at.selectbox(key="w_half_" + lang).set_value("bottom").run()
at.selectbox(key="w_outs").set_value(2).run()
at.checkbox(key="w_r1").uncheck().run()
at.checkbox(key="w_r2").check().run()
at.checkbox(key="w_r3").uncheck().run()
at.slider(key="w_diff").set_value(-3).run()
at.selectbox(key="w_strikes").set_value(0).run()
at.selectbox(key="w_balls").set_value(0).run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
no_pay = (("どの成功率でもこの盗塁は得になりません", "割に合いません") if lang == "JA" else
          ("No success rate makes this steal pay off", "no success rate makes this steal pay off"))
assert md.count(no_pay[0]) == 1 and no_pay[1] not in md, md[-600:]
assert ("牽制や暴投" if lang == "JA" else "pickoff and wild-pitch") in md

# postseason rules: without the automatic runner, top 10 tied differs
at.selectbox(key="w_inning").set_value(10).run()
at.selectbox(key="w_half_" + lang).set_value("top").run()
at.checkbox(key="w_r1").uncheck().run()
at.checkbox(key="w_r2").uncheck().run()
at.slider(key="w_diff").set_value(0).run()
at.selectbox(key="w_balls").set_value(0).run()
at.selectbox(key="w_outs").set_value(0).run()
import analysis
def gauge_md():
    return " ".join(m.value for m in at.markdown)
with_runner = analysis.batting_wp(analysis.home_wp(10, "top", 0, (0, 0, 0), 0), "top")
without = analysis.batting_wp(analysis.home_wp(10, "top", 0, (0, 0, 0), 0, extra_runner=False), "top")
assert f"{with_runner * 100:.1f}%" in gauge_md(), gauge_md()[-400:]
at.checkbox(key="w_xr").uncheck().run()
assert not at.exception, at.exception
assert f"{without * 100:.1f}%" in gauge_md()
at.checkbox(key="w_xr").check().run()

# a live state applied to the widgets (what "Apply to Analysis" stores)
at.session_state["_preset"] = {"inning": 11, "top_bottom": "bottom", "outs": 2,
                               "runners": (0, 1, 0), "score_diff": -14, "balls": 3,
                               "strikes": 2, "sched": 9, "extra_runner": False}
at.run()
assert not at.exception, at.exception
assert at.checkbox(key="w_xr").value is False
assert at.selectbox(key="w_inning").value == 11 and at.slider(key="w_diff").value == -10
assert any("-14" in c.value for c in at.caption), [c.value for c in at.caption]

# live panel: auto-refresh fetches again once its interval has passed, without a click
at.button(key="fetch_games").click().run()
at.button(key="load_live").click().run()
assert not at.exception, at.exception
assert len(fetches) == 1 and any(b.key == "apply_live" for b in at.button)
import analysis
_t = analysis.team_layer_wp(5, "top", 1, (1, 0, 0), 1, 9, True, 1, 2, "2026-07-01", "R")
assert any(TEAM_WORD in c.value and f"{_t['team_wp'] * 100:.1f}%" in c.value
           and f"{_t['engine_wp'] * 100:.1f}%" in c.value for c in at.caption),     [c.value for c in at.caption]
assert fragments == []                         # auto-refresh off: no timer
at.checkbox(key="auto_refresh").check().run()
assert not at.exception, at.exception
assert fragments and fragments[-1] == 30, fragments   # the 30 s timer is armed
assert len(fetches) == 1                       # just loaded: no second fetch yet
assert any("UTC" in c.value for c in at.caption), [c.value for c in at.caption]
at.session_state["live_last_refresh"] = 0.0     # as if 30 s had passed
LIVE["status"] = "Final"
at.run()
assert not at.exception, at.exception
assert len(fetches) == 2
assert not any(b.key == "apply_live" for b in at.button)   # game over: nothing to apply
# a failed refresh keeps the card and says so
live_feed.get_live_state = lambda pk: fetches.append(pk)
at.session_state["live_last_refresh"] = 0.0
at.run()
assert not at.exception, at.exception
assert len(fetches) == 3
assert any(("failed" if lang == "EN" else "失敗") in c.value for c in at.caption), \
    [c.value for c in at.caption]
live_feed.get_live_state = fake_state
at.checkbox(key="auto_refresh").uncheck().run()
at.session_state["live_last_refresh"] = 0.0
n_frag = len(fragments)
at.run()
assert len(fetches) == 3                       # off: no refetch
assert len(fragments) == n_frag                # and no timer

# a postseason game: the team layer is not used, and the card says so
LIVE.update({"status": "In Progress", "game_type": "F"})
at.button(key="load_live").click().run()
assert not at.exception, at.exception
cap = [c.value for c in at.caption]
assert not any(TEAM_WORD in c for c in cap), cap
assert any(("regular-season" if lang == "EN" else "レギュラーシーズン") in c for c in cap), cap

# replay mode renders without network until a button is pressed
at.sidebar.radio[1].set_value("🎬 Replay Mode").run()
assert not at.exception, at.exception
# a replayed regular-season play shows the team layer for the state before it
[b for b in at.button if b.label == "Load Games"][0].click().run()
[b for b in at.button if b.label == "▶ Load Replay"][0].click().run()
assert not at.exception, at.exception
_r = analysis.team_layer_wp(6, "bottom", 0, (0, 0, 0), -1, 9, True, 1, 2, "2026-07-01", "R")
assert any(TEAM_WORD in c.value and f"{_r['team_wp'] * 100:.1f}%" in c.value for c in at.caption),     [c.value for c in at.caption]
print("OK", lang, "key" if with_key else "nokey")
"""


@pytest.mark.parametrize("lang,key", [("EN", "nokey"), ("JA", "nokey"), ("EN", "key"), ("JA", "key")])
def test_app_renders(lang, key):
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    if key == "key":
        env["ANTHROPIC_API_KEY"] = "test-not-a-key"
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code, lang, key], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert f"OK {lang} {key}" in r.stdout
