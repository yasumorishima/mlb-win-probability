"""Late-inning bullpen by usage (pen_usage.py, bench.py pen_avail_*)."""

import json
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bench_live as BL  # noqa: E402
import bench_replay as R  # noqa: E402
import game_fixture as F  # noqa: E402
import pen_usage as PU  # noqa: E402
from game import D, MAX_DIFF  # noqa: E402

RAW = json.loads((PU.PUB / "bullpen_usage_2026.json").read_text())


def test_usage_window_ends_the_day_before():
    t, d0 = RAW["late"][len(RAW["late"]) // 2][:2]
    day = date.fromisoformat(d0)
    exp = {}
    for tt, dd, p, ii, ls, n in RAW["late"]:
        x = date.fromisoformat(dd)
        if tt == t and day - timedelta(days=30) <= x < day:
            exp.setdefault(p, np.zeros((3, 4)))[ii, ls] += n
    got = PU.usage(t, day)
    assert set(got) == set(exp) and all(np.array_equal(got[p], exp[p]) for p in exp)
    today = [r for r in RAW["late"] if r[0] == t and r[1] == d0]
    nxt = PU.usage(t, day + timedelta(days=1))
    p, ii, ls, n = today[0][2], today[0][3], today[0][4], today[0][5]
    assert nxt[p][ii, ls] >= n                            # the day itself counts only from the next day


def test_tired_means_both_previous_days():
    rel = {}
    for t, d, p in RAW["relief"]:
        rel.setdefault((t, d), set()).add(p)
    for (t, d), ps in rel.items():
        day = date.fromisoformat(d)
        prev = rel.get((t, (day - timedelta(days=1)).isoformat()), set())
        both = ps & prev
        if both:
            assert both <= PU.tired(t, day + timedelta(days=1))
            assert not (ps - prev) & PU.tired(t, day + timedelta(days=1))
            return
    raise AssertionError("no back-to-back relievers found")


FEED = F.game_feed()
TL = R.timeline(FEED)
K_PC = next(t["k"] for t in TL if any(d["type"] == "pitch_change" for d in TL[t["k"]]["decisions"]))


def _moment():
    m, data, ex = R.moment_at(FEED, K_PC)
    m = replace(m, inning=7, cur_is_starter=False)
    return m, data, ex


def test_empty_usage_equals_pooled_bullpen():
    m, data, ex = _moment()
    b = BL._bench(True)
    data.pen_avail_fld = data.pen_avail_bat = None
    pooled = b.options(m, data, pinch_hitters=ex["bench"][:2], relievers=ex["bullpen"][:3])
    data.pen_avail_fld, data.pen_avail_bat = {}, {}
    empty = b.options(m, data, pinch_hitters=ex["bench"][:2], relievers=ex["bullpen"][:3])
    assert max(abs(x["wp_offense"] - y["wp_offense"]) for x, y in zip(pooled, empty)) < 1e-12


def _closer_case():
    m, data, ex = _moment()
    closer, other = F.RP[2], F.RP[3]
    pit = dict(data.pit)
    pit[closer] = np.array([400, 20, 60, 15, 1, 5, 500], float)          # a dominant arm (K BB 1B 2B 3B HR OUT)
    data = replace(data, pit=pit)
    u_c = np.zeros((3, 4)); u_c[2, 2] = 60                                  # 9th inning, ahead by 1-3
    u_o = np.zeros((3, 4)); u_o[2, 0] = 60                                  # 9th inning, behind
    data.pen_avail_fld = {closer: u_c, other: u_o}
    data.pen_avail_bat = {}
    return m, data, closer


def test_using_the_closer_now_costs_the_ninth():
    m, data, closer = _closer_case()                   # bottom half: the away team fields
    b = BL._bench(True)
    b._cache = {}
    keep = b.later_halves(m, data, m.lineup)
    gone = b.later_halves(m, data, m.lineup, exclude=(closer,))
    up_1 = MAX_DIFF + (-1)                              # home - away = -1: the away team leads by one
    down_1 = MAX_DIFF + 1                               # the away team trails by one
    cost_lead = gone[(9, "bottom")][up_1] - keep[(9, "bottom")][up_1]
    cost_trail = gone[(9, "bottom")][down_1] - keep[(9, "bottom")][down_1]
    assert cost_lead > 0.005                            # home WP rises without the closer when the away team leads
    assert cost_lead > 3 * abs(cost_trail)              # ... far more than when it trails (he pitches with a lead)


def test_pen_plan_names_the_closer_for_the_ninth():
    m, data, closer = _closer_case()
    m = replace(m, diff=-2)                             # away (fielding) leads by 2
    plan = BL.pen_plan(m, data, {closer: "CLOSER"})
    ninth = next(r for r in plan if r["inning"] == 9)
    assert ninth["rows"][0][0] == "CLOSER"
    assert abs(sum(x for _, x in ninth["rows"]) + ninth["other"] - 1) < 1e-9


def test_lead_state_of_the_fielding_team():
    import bench as B
    at = lambda d: MAX_DIFF + d                         # index of home - away = d
    # top half: the home team fields, its lead = home - away
    assert B.PEN_LEAD["top"][at(2)] == 2 and B.PEN_LEAD["top"][at(-1)] == 0 and B.PEN_LEAD["top"][at(5)] == 3
    # bottom half: the away team fields, its lead = away - home
    assert B.PEN_LEAD["bottom"][at(-1)] == 2 and B.PEN_LEAD["bottom"][at(1)] == 0 and B.PEN_LEAD["bottom"][at(0)] == 1
