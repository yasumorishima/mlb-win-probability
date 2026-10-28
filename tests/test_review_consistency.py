"""Look-back: the moment card and the game summary rate a call on the same moment
and show the same WP (offline)."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bench_live as BL  # noqa: E402
import bench_replay as R  # noqa: E402
import game_fixture as F  # noqa: E402

FEED = F.game_feed()
K_PC = next(t["k"] for t in R.timeline(FEED) if any(d["type"] == "pitch_change" for d in t["decisions"]))


def _ph_after_change():
    """The away team changes pitchers (event 0), then the home team sends up a pinch hitter
    for the batter due up (event 1)."""
    feed = copy.deepcopy(FEED)
    plays = [p for p in feed["liveData"]["plays"]["allPlays"] if p["about"]["isComplete"]]
    m, _, _ = R.moment_at(feed, K_PC)
    slot = m.slot
    p = plays[K_PC]
    p["playEvents"].append({"details": {"eventType": "offensive_substitution", "description": "Pinch-hitter X"},
                            "player": {"id": F.HOME_BENCH[1]}, "battingOrder": f"{slot + 1}01",
                            "position": {"abbreviation": "PH"}})
    return feed


def test_upto_only_after_an_earlier_substitution():
    feed = _ph_after_change()
    t = R.timeline(feed)[K_PC]
    pc = next(d for d in t["decisions"] if d["type"] == "pitch_change")
    ph = next(d for d in t["decisions"] if d["type"] == "pinch")
    assert R.upto_for(t, pc) is None and R.upto_for(t, ph) == ph["idx"]
    assert R.upto_for(t, {"type": "steal"}) is None


def test_summary_rates_a_call_at_the_moment_it_was_made():
    feed = _ph_after_change()
    rows = R.game_review(feed)
    row = next(r for r in rows if r["k"] == K_PC and r["type"] == "pinch")
    t = R.timeline(feed)[K_PC]
    ph = next(d for d in t["decisions"] if d["type"] == "pinch")
    want = R.rate(ph, BL.bench_options(feed, got=R.moment_at(feed, K_PC, upto_event=ph["idx"])))
    before = R.rate(ph, BL.bench_options(feed, got=R.moment_at(feed, K_PC)))
    assert row["upto"] == ph["idx"]
    assert row["rating"] is not None and want is not None
    assert abs(row["rating"]["gain"] - want["gain"]) < 1e-12
    assert before is None or abs(before["gain"] - want["gain"]) > 1e-9     # the new pitcher matters here
    # the pitching change itself is still rated before it happened
    pcr = next(r for r in rows if r["k"] == K_PC and r["type"] == "pitch_change")
    pc = next(d for d in t["decisions"] if d["type"] == "pitch_change")
    want_pc = R.rate(pc, BL.bench_options(feed, got=R.moment_at(feed, K_PC)))
    assert pcr["upto"] is None and abs(pcr["rating"]["gain"] - want_pc["gain"]) < 1e-12


def test_summary_swing_uses_the_same_series_as_the_card(monkeypatch):
    import bench_review as BR
    import pen_wp
    tl = R.timeline(FEED)
    fake = {t["at_bat_index"]: 0.5 + 0.01 * (t["k"] % 7) for t in tl}
    monkeypatch.setattr(pen_wp, "game", lambda pk, d: fake)
    monkeypatch.setattr(BR, "_feed", lambda pk: FEED)
    series = BR._series(777, FEED, tl)
    assert all(abs(series[t["k"]]["wp"] - fake[t["at_bat_index"]]) < 1e-12 for t in tl)
    BR._review.clear()
    rows = BR._review(777)
    BR._review.clear()
    for r in rows:
        k = r["k"]
        if k + 1 < len(tl):
            a, b = fake[tl[k]["at_bat_index"]], fake[tl[k + 1]["at_bat_index"]]
            want = 100 * ((b - a) if r["side"] == "home" else (a - b))
            assert abs(r["swing"] - want) < 1e-9, (r, want)


CHILD = r"""
import sys
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer, pen_wp
import bench_replay as R, bench_review as BR
import test_review_consistency as T
FEED = T._ph_after_change()
live_feed.get_todays_games = lambda date=None: [{"gamePk": 777, "away_team": "Away", "home_team": "Home",
                                                 "status": "Final", "start_time_utc": ""}]
team_layer.strengths = lambda h, a, d: {"d_prev": 0.1, "d_cur": 0.0, "teams": {}}
bench_live.get_feed = lambda pk: FEED
pen_wp.game = lambda pk, d: {}
t = R.timeline(FEED)[T.K_PC]
ph = next(d for d in t["decisions"] if d["type"] == "pinch")
x = BR.TXT["JA"]
def txt(o):
    r = R.rate(ph, o)
    return x["rated"].format(d=r["gain"], n=r["n"], r=r["rank"])
after = txt(bench_live.bench_options(FEED, got=R.moment_at(FEED, T.K_PC, upto_event=ph["idx"])))
before = txt(bench_live.bench_options(FEED, got=R.moment_at(FEED, T.K_PC)))
assert after != before, (after, before)
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("JA").run()
at.sidebar.radio[1].set_value("📋 Bench").run()
at.radio(key="bench_view_JA").set_value("振り返り").run()
sel = [s for s in at.selectbox if s.key and s.key.startswith("review_dec_")][0]
sel.set_value(T.K_PC).run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert after in md and before not in md and x["after_sub"] in md, md[-1500:]
print("OK")
"""


def test_card_rates_a_call_at_the_moment_it_was_made():
    import os
    import subprocess
    import pytest
    pytest.importorskip("streamlit")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    code = "ROOT = " + repr(str(ROOT)) + "\n" + CHILD
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "OK" in r.stdout
