"""Pre-generated game notes in the look-back view (offline)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/game_notes"))


def test_note_lookup(tmp_path, monkeypatch):
    import bench_review as BR
    (tmp_path / "notes_2026.json").write_text(json.dumps({"notes": {"777": "テストの解説です。"}}, ensure_ascii=False))
    monkeypatch.setattr(BR, "NOTES_DIR", tmp_path)
    BR._notes.clear()
    assert BR.game_note(777, "2026-07-01") == "テストの解説です。"
    assert BR.game_note(778, "2026-07-01") is None and BR.game_note(777, "2025-07-01") is None
    assert BR.game_note(777, None) is None
    BR._notes.clear()


def _obj():
    return {"game_pk": 1, "home": "Home Team", "away": "Away Team", "final": "Away Team 2 - 5 Home Team",
            "home_wp_start_pct": 54, "home_wp_min_pct": 40, "home_wp_max_pct": 100,
            "big_plays": [{"batter": "A Batter", "pitcher": "P Pitcher", "score_before": "Away Team 2 - 2 Home Team",
                           "score_after": "Away Team 2 - 5 Home Team", "batting_team_wp_before_pct": 48,
                           "batting_team_wp_after_pct": 86, "runs_scored": 3}],
            "late_pitching_changes": [{"new_pitcher": "R Reliever", "score": "Away Team 2 - 5 Home Team", "team_wp_pct": 14}]}


def test_checker_rejects_numbers_scores_and_percents_not_in_the_input():
    import check_notes as C
    pad = "終盤は R Reliever が登板し、そのまま試合が終わりました。最後まで落ち着いた試合運びでした。"
    ok = f"Home Team が 5-2 で勝ちました。A Batter の一打で勝率は 48% から 86% に上がりました。{pad}"
    assert C.verdict(_obj(), {"stop_reason": "end_turn", "text": ok}) == "ok"
    assert C.verdict(_obj(), {"stop_reason": "end_turn", "text": ok.replace("5-2", "5-3")}) == "score_not_in_input"
    assert C.verdict(_obj(), {"stop_reason": "end_turn", "text": ok.replace("48%", "5%")}) == "percent_not_in_input"
    assert C.verdict(_obj(), {"stop_reason": "end_turn", "text": ok.replace("48%", "47%")}) == "number_not_in_input"
    assert C.verdict(_obj(), {"stop_reason": "max_tokens", "text": ok}) == "stop_reason"
    assert C.verdict(_obj(), {"stop_reason": "end_turn", "text": "短い。"}) == "length"


CHILD = r"""
import sys, json, tempfile, pathlib
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tests")
from streamlit.testing.v1 import AppTest
import live_feed, bench_live, team_layer, pen_wp
import game_fixture as F
FEED = F.game_feed()
live_feed.get_todays_games = lambda date=None: [{"gamePk": 777, "away_team": "Away", "home_team": "Home",
                                                 "status": "Final", "start_time_utc": ""}]
team_layer.strengths = lambda h, a, d: {"d_prev": 0.1, "d_cur": 0.0, "teams": {}}
bench_live.get_feed = lambda pk: FEED
pen_wp.game = lambda pk, d: {}
d = pathlib.Path(tempfile.mkdtemp())
(d / "notes_2026.json").write_text(json.dumps({"notes": {"777": "ホームが 1-0 で勝った試合の解説です。"}}, ensure_ascii=False))
import bench_review                       # after the patches above (it imports get_todays_games by name)
bench_review.NOTES_DIR = d
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("JA").run()
at.sidebar.radio[1].set_value("📋 Bench").run()
at.radio(key="bench_view_JA").set_value("振り返り").run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "1-0 で勝った試合の解説" in md, md[:800]
print("OK")
"""


def test_note_is_shown_in_the_look_back_view():
    import os
    import subprocess
    import pytest
    pytest.importorskip("streamlit")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    r = subprocess.run([sys.executable, "-c", "ROOT = " + repr(str(ROOT)) + "\n" + CHILD], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
