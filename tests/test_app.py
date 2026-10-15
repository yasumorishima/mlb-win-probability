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

    def fake_advise(situation, engine, players):
        adv = Advice(recommendation="intentional_walk", side="defense", confidence=0.55,
                     follows_engine=True, reason="<b>Walk him</b>; first base is open.",
                     departure_reason="",
                     cited=[Citation(name="leverage_index", value=engine["leverage_index"])])
        return adv, ["made-up problem for the test"]

    strategist.advise = fake_advise

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
assert at.selectbox(key="w_half").value == "bottom"
assert at.checkbox(key="w_r1").value and at.checkbox(key="w_r2").value
assert at.slider(key="w_diff").value == -1
md = " ".join(m.value for m in at.markdown)
assert ("送りバント" if lang == "JA" else "Sacrifice bunt") in md

# the preset values survive another interaction (widgets keep them)
at.selectbox(key="w_balls").set_value(3).run()
assert not at.exception, at.exception
assert at.selectbox(key="w_inning").value == 7

if with_key:
    at.button(key="ask_coach").click().run()
    assert not at.exception, at.exception
    md = " ".join(m.value for m in at.markdown)
    assert ("敬遠" if lang == "JA" else "Intentional walk") in md
    assert "&lt;b&gt;Walk him&lt;/b&gt;" in md          # model text is escaped
    assert any("made-up problem" in w.value for w in at.warning)

# replay mode renders without network until a button is pressed
at.sidebar.radio[1].set_value("🎬 Replay Mode").run()
assert not at.exception, at.exception
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
