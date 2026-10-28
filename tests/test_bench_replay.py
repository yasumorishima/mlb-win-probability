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


def test_rating_of_each_call():
    import bench_viz as V
    series = V.wp_series(FEED)
    rows = R.game_review(FEED, series)
    by = {r["type"]: r for r in rows}
    assert {"pinch", "pitch_change", "steal"} <= set(by)
    assert by["pinch"]["side"] == "home" and by["pitch_change"]["side"] == "away" and by["steal"]["side"] == "home"
    # pinch hitter: the PH option at that moment, ranked among pinch hitters only
    k = by["pinch"]["k"]
    opts = BL.bench_options(FEED, got=R.moment_at(FEED, k))
    ph = sorted([o for o in opts["all"] if o["kind"] == "pinch_hit"], key=lambda o: -o["wp_offense"])
    me = next(o for o in ph if o["option"] == f"pinch_hit:{F.PH}")
    rt = by["pinch"]["rating"]
    assert rt["me"]["kind"] == "pinch_hit" and rt["n"] == len(ph) and rt["rank"] == ph.index(me) + 1
    assert abs(rt["gain"] - 100 * me["delta_wp_offense"]) < 1e-9
    # pitching change: rated from the fielding team's side (sign flipped)
    k2 = by["pitch_change"]["k"]
    opts2 = BL.bench_options(FEED, got=R.moment_at(FEED, k2))
    pc = next(o for o in opts2["all"] if o["option"] == f"pitch_change:{F.RP[2]}")
    assert abs(by["pitch_change"]["rating"]["gain"] + 100 * pc["delta_wp_offense"]) < 1e-9
    best = min((o for o in opts2["all"] if o["kind"] == "pitch_change"), key=lambda o: o["wp_offense"])
    assert by["pitch_change"]["rating"]["best"]["option"] == best["option"]
    # swing over the play, for the team that made the call
    assert abs(by["pinch"]["swing"] - 100 * (series[k + 1]["wp"] - series[k]["wp"])) < 1e-9
    assert abs(by["pitch_change"]["swing"] - 100 * (series[k2]["wp"] - series[k2 + 1]["wp"])) < 1e-9


def test_pinch_runner_is_rated_as_a_pinch_runner():
    ph = {"kind": "pinch_hit", "option": "pinch_hit:5", "wp_offense": 0.6, "delta_wp_offense": 0.1}
    pr = [{"kind": "pinch_run", "option": "pinch_run:5", "wp_offense": 0.52, "delta_wp_offense": 0.02},
          {"kind": "pinch_run", "option": "pinch_run:6", "wp_offense": 0.51, "delta_wp_offense": 0.01}]
    opts = {"all": [ph] + pr}
    r = R.rate({"type": "pinch", "player": 5, "pos": "PR", "text": "Pinch-runner X replaces Y"}, opts)
    assert r["me"]["option"] == "pinch_run:5" and r["n"] == 2 and r["rank"] == 1
    r = R.rate({"type": "pinch", "player": 5, "pos": "PH", "text": "Pinch-hitter X replaces Y"}, {"all": pr + [ph]})
    assert r["me"]["option"] == "pinch_hit:5" and r["n"] == 1
    assert R.rate({"type": "pinch", "player": 9, "pos": "PH", "text": ""}, opts) is None


def test_missed_chances_skip_calls_that_were_made():
    rows = R.missed_chances(FEED, min_gain=-1e9, top=10_000)
    tl = {t["k"]: t for t in TL}
    assert len(rows) == len(TL)                         # with no threshold every moment has a best move
    for r in rows:
        t = tl[r["k"]]
        made = {(R.deciding_side(t, d), d["type"]) for d in t["decisions"]}
        assert (r["side"], R.MISSED_KINDS[r["option"]["kind"]]) not in made
        assert r["option"]["kind"] != "pitch_change"     # the fixture has 4 innings: no changes before the 7th
    assert all(r["gain"] >= 1.0 for r in R.missed_chances(FEED))


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
import pen_wp
pen_wp.game = lambda pk, d: {i: 0.123 for i in range(200)}     # PEN_RECAL home WP for every plate appearance
at = AppTest.from_file(ROOT + "/streamlit_app.py", default_timeout=180)
at.run()
at.sidebar.radio[0].set_value("JA").run()
at.sidebar.radio[1].set_value("📋 Bench").run()
at.radio(key="bench_view_JA").set_value("振り返り").run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "実際には" in md and "勝率" in md
assert "継投の使い方込みの勝率" in md and (">12%<" in md or ">88%<" in md), md[-1500:]     # the look-back WP is PEN_RECAL
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
# the game summary
at.radio(key="review_view_JA").set_value("📝 試合のまとめ").run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "Home：采配" in md and "Away：采配" in md, md[-800:]
df = at.dataframe[0].value
assert len(df) == len([d for t in tl for d in t["decisions"] if d["type"] != "defense"]), df
assert any("代打" in str(v) or "代走" in str(v) for v in df[df.columns[2]])
at.button(key="review_miss_777").click().run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "モデルが 1 ポイント以上" in md
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


def test_position_player_who_pitched_later_is_not_a_reliever():
    import copy
    feed = copy.deepcopy(FEED)
    x = F.AWAY_BENCH[0]
    feed["liveData"]["boxscore"]["teams"]["away"]["players"][f"ID{x}"]["position"] = {"abbreviation": "P"}   # end-of-game position
    feed["gameData"]["players"][f"ID{x}"]["primaryPosition"] = {"abbreviation": "RF"}
    _, _, ex = R.moment_at(feed, K_PC)                  # bottom half: the away team fields
    assert x not in ex["bullpen"]
    k_top = next(t["k"] for t in TL if t["top_bottom"] == "top")
    _, _, ex2 = R.moment_at(feed, k_top)                # top half: the away team bats, he is on its bench
    assert x in ex2["bench"]


def test_call_after_an_earlier_substitution_in_the_same_plate_appearance():
    import copy
    feed = copy.deepcopy(FEED)
    plays = [p for p in feed["liveData"]["plays"]["allPlays"] if p["about"]["isComplete"]]
    p = plays[K_PC]                                     # the away team changes pitchers here (event 0)
    p["playEvents"].append({"details": {"eventType": "offensive_substitution", "description": "Pinch-hitter X"},
                            "player": {"id": F.HOME_BENCH[1]}, "battingOrder": "301", "position": {"abbreviation": "PH"}})
    tl = R.timeline(feed)
    ph = next(d for d in tl[K_PC]["decisions"] if d["type"] == "pinch")
    m0, _, _ = R.moment_at(feed, K_PC)
    m1, _, ex1 = R.moment_at(feed, K_PC, upto_event=ph["idx"])
    assert m0.pitcher == F.SP[1] and m1.pitcher == F.RP[2]          # the pinch hitter faces the new pitcher
    assert not m1.cur_is_starter
    pc = next(d for d in tl[K_PC]["decisions"] if d["type"] == "pitch_change")
    m2, _, _ = R.moment_at(feed, K_PC, upto_event=pc["idx"])
    assert m2.pitcher == F.SP[1]                                     # the change itself: decided before it happened
