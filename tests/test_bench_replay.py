"""bench_replay / bench_review: looking back at a finished game (offline)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bench_live as BL  # noqa: E402
import bench_replay as R  # noqa: E402
import game_fixture as F  # noqa: E402

FEED = F.game_feed()
TL = R.timeline(FEED)
K_PH = next(t["k"] for t in TL if any(d["type"] == "pinch" for d in t["decisions"]))
K_PC = next(t["k"] for t in TL if any(d["type"] == "pitch_change" for d in t["decisions"]))


def test_timeline_reads_the_bench_decisions():
    kinds = {d["type"] for t in TL for d in t["decisions"]}
    assert {"pinch", "pitch_change", "steal"} <= kinds
    assert TL[K_PH]["inning"] == 3 and TL[K_PH]["top_bottom"] == "bottom"


def test_lineup_and_bench_as_they_stood():
    m, data, ex = R.moment_at(FEED, K_PH)
    assert ex["batter"] == F.HOME_LU[7]                 # before the pinch hitter: the original batter is due
    assert F.PH in ex["bench"]                          # ... and the pinch hitter is still on the bench
    m2, _, ex2 = R.moment_at(FEED, K_PH + 1)
    assert m2.lineup[7] == F.PH and F.PH not in ex2["bench"]   # afterwards he is in the lineup
    m3, _, ex3 = R.moment_at(FEED, K_PC)
    assert m3.pitcher == F.SP[1]                        # before the change: the starter is still pitching
    assert F.RP[2] in ex3["bullpen"] or F.RP[2] not in BL._snapshot(BL.snapshot_for_season(2026))["role"]
    _, _, ex4 = R.moment_at(FEED, K_PC + 1)
    assert F.RP[2] not in ex4["bullpen"]                # once used, not offered again


def test_no_hindsight_records():
    _, _, ex = R.moment_at(FEED, 0)
    assert ex["records_through"].startswith("2025")     # a 2026 game uses records that end with 2025


def test_runner_and_options_at_a_steal_moment():
    k = next(t["k"] for t in TL if any(d["type"] == "steal" for d in t["decisions"]))
    got = R.moment_at(FEED, k)
    m = got[0]
    assert m.bases == (1, 0, 0) and m.runner_ids[0] is not None
    out = BL.bench_options(FEED, got=got)
    assert any(o["kind"] == "steal" for o in out["offense"])


CHILD = r"""
import sys
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer
import game_fixture as F
FEED = F.game_feed()
live_feed.get_todays_games = lambda date=None: [{"gamePk": 777, "away_team": "Away", "home_team": "Home",
                                                 "status": "Final", "start_time_utc": ""}]
team_layer.strengths = lambda h, a, d: {"d_prev": 0.1, "d_cur": 0.0, "teams": {}}
bench_live.get_feed = lambda pk: FEED
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("JA").run()
at.sidebar.radio[1].set_value("📋 Bench").run()
at.radio(key="bench_view_JA").set_value("振り返り").run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "実際には" in md and "勝率" in md
# jump to the pinch hitter through the decisions list
sel = [s for s in at.selectbox if s.key and s.key.startswith("review_dec_")][0]
import bench_replay
tl = bench_replay.timeline(FEED)
kph = next(t["k"] for t in tl if any(d["type"] == "pinch" for d in t["decisions"]))
sel.set_value(kph).run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "代打・代走" in md and "Pinch-hitter" in md, md[-800:]
assert "モデルの評価" in md, md[-800:]
# next / previous buttons move the moment
at.button(key="review_next").click().run()
assert not at.exception, at.exception
assert at.session_state["review_k"] == kph + 1
at.button(key="review_hot").click().run()
assert not at.exception, at.exception
print("OK")
"""


def test_review_screen():
    pytest.importorskip("streamlit")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "OK" in r.stdout
