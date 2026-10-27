"""bench_live: reading the bench's moment from a live feed and scoring options (offline)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_live as BL  # noqa: E402

SNAP = BL._snapshot()
BATTERS = sorted(SNAP["_bat"], key=lambda i: -SNAP["_bat"][i].sum())[:30]          # well-known batters
RELIEVERS = [int(k) for k, (st, g) in SNAP["role"].items() if st == 0 and g >= 40][:6]
STARTERS = [int(k) for k, (st, g) in SNAP["role"].items() if st >= 20 and st == g][:4]


def feed(inning=7, top=False, outs=1, first=True, second=False, home=3, away=3, bench=None, bullpen=None):
    home_lu, away_lu = BATTERS[:9], BATTERS[9:18]
    bat_lu = away_lu if top else home_lu
    team = {"home": {"id": 121}, "away": {"id": 143}}
    bat_side, fld_side = ("away", "home") if top else ("home", "away")
    box = {
        bat_side: {"team": team[bat_side], "battingOrder": bat_lu, "bench": bench if bench is not None else BATTERS[18:22],
                   "bullpen": RELIEVERS[3:6], "pitchers": [STARTERS[0]]},
        fld_side: {"team": team[fld_side], "battingOrder": home_lu if top else away_lu, "bench": [],
                   "bullpen": bullpen if bullpen is not None else RELIEVERS[:3] + STARTERS[1:3], "pitchers": [STARTERS[3]]},
    }
    offense = {"batter": {"id": bat_lu[4]}, "team": team[bat_side]}
    if first:
        offense["first"] = {"id": 1}
    if second:
        offense["second"] = {"id": 2}
    return {
        "gameData": {"status": {"detailedState": "In Progress"}, "game": {"type": "R", "season": "2027"},
                     "teams": {"home": {"id": 121, "name": "Home"}, "away": {"id": 143, "name": "Away"}},
                     "players": {}},
        "liveData": {"linescore": {"teams": {"home": {"runs": home}, "away": {"runs": away}}, "offense": offense,
                                   "defense": {"batter": {"id": (home_lu if top else away_lu)[2]}, "team": team[fld_side]},
                                   "scheduledInnings": 9},
                     "plays": {"currentPlay": {"about": {"inning": inning, "isTopInning": top, "isComplete": False},
                                               "count": {"balls": 0, "strikes": 0, "outs": outs}}},
                     "boxscore": {"teams": box}},
    }


def test_moment_read_from_feed():
    m, data, extra = BL.moment_from_feed(feed())
    assert (m.inning, m.half, m.outs, m.bases, m.diff) == (7, "bottom", 1, (1, 0, 0), 0)
    assert m.slot == 4 and m.fld_next == 2
    assert not set(extra["bullpen"]) & set(STARTERS)          # starters are not offered as relievers
    assert len(extra["bullpen"]) == 3
    # season-decayed records and the matching shrinkage (size_decay.py) are used
    import layer as Ly
    assert data.prm["kb1"] == pytest.approx(0.7 * Ly.PARAMS["kb1"]) and data.prm["kd"] == Ly.PARAMS["kd"]
    b = m.lineup[0]
    assert data.bat[b].sum() < SNAP["_bat_all"][b].sum()


def test_options_shape_and_signs():
    out = BL.bench_options(feed())
    kinds = [o["kind"] for o in out["offense"]]
    assert kinds[0] == "none" and "bunt" in kinds and kinds.count("pinch_hit") == 4
    assert [o["kind"] for o in out["defense"]][0] == "intentional_walk"
    none = out["offense"][0]
    assert none["delta_wp_offense"] == 0.0 and 0 < none["wp_offense"] < 1 and 0 < none["p_score"] < 1
    ph = [o["wp_offense"] for o in out["offense"] if o["kind"] == "pinch_hit"]
    assert ph == sorted(ph, reverse=True)                     # best pinch hitter first
    rp = [o["wp_offense"] for o in out["defense"] if o["kind"] == "pitch_change"]
    assert rp == sorted(rp)                                   # best reliever (lowest offense WP) first


def test_no_bunt_with_two_outs_or_bases_empty():
    assert "bunt" not in [o["kind"] for o in BL.bench_options(feed(outs=2))["offense"]]
    assert "bunt" not in [o["kind"] for o in BL.bench_options(feed(first=False))["offense"]]


def test_unreadable_moment_returns_none():
    f = feed()
    f["liveData"]["boxscore"]["teams"]["home"]["battingOrder"] = BATTERS[:8]
    assert BL.bench_options(f) is None


def test_pinch_hitter_keeps_the_slot_for_later_innings():
    """A pinch hitter changes the later halves too: with the same player as
    the only option, the value equals 'none'."""
    f = feed(bench=[BATTERS[4]])                              # the current batter himself
    out = BL.bench_options(f)
    none, ph = out["offense"][0], [o for o in out["offense"] if o["kind"] == "pinch_hit"][0]
    assert ph["wp_offense"] == pytest.approx(none["wp_offense"], abs=1e-12)


def test_postseason_has_no_automatic_runner():
    """Tied in the 9th: without the automatic runner the extra innings differ,
    so the offense WP differs; the gate holds for both rules."""
    reg = feed(inning=9, top=True, outs=0, home=2, away=2)
    post = feed(inning=9, top=True, outs=0, home=2, away=2)
    post["gameData"]["game"]["type"] = "W"
    a = BL.bench_options(reg)["offense"][0]["wp_offense"]
    b = BL.bench_options(post)["offense"][0]["wp_offense"]
    assert abs(a - b) > 1e-4, (a, b)


FAST = sorted(((v, int(k)) for k, v in SNAP["speed"].items() if int(k) in SNAP["_bat"]), reverse=True)


def test_steal_call_and_pinch_runner():
    """Runner on first: a steal row with the runner's estimated success and
    the break-even; faster pinch runners help the offense, best first."""
    f = feed(bench=[FAST[0][1], FAST[1][1], BATTERS[20]])
    out = BL.bench_options(f)
    steal = [o for o in out["offense"] if o["kind"] == "steal"]
    assert len(steal) == 1 and 0 < steal[0]["breakeven"] < 1 and 0 < steal[0]["p_success"] < 1
    assert steal[0]["go"] == (steal[0]["p_success"] > steal[0]["breakeven"])
    pr = [o for o in out["offense"] if o["kind"] == "pinch_run"]
    assert pr and all(o["delta_wp_offense"] > 0 for o in pr), pr
    assert [o["wp_offense"] for o in pr] == sorted([o["wp_offense"] for o in pr], reverse=True)


def test_faster_runner_steals_more_often():
    import bench as B
    m, data, _ = BL.moment_from_feed(feed())
    slow = dict(data.speed); slow[1] = -2.0
    fast = dict(data.speed); fast[1] = 2.0
    b = BL._bench(True)
    p_slow = [o for o in b.options(m, B.Data(**{**data.__dict__, "speed": slow})) if o["option"].startswith("steal")][0]
    p_fast = [o for o in b.options(m, B.Data(**{**data.__dict__, "speed": fast})) if o["option"].startswith("steal")][0]
    assert p_fast["p_success"] > p_slow["p_success"]
    assert p_fast["breakeven"] == pytest.approx(p_slow["breakeven"])
