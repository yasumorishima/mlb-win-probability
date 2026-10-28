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
    u_c = np.zeros((3, 5)); u_c[2, 3] = 60                                  # 9th inning, ahead by 1-3 (5 lead states)
    u_o = np.zeros((3, 5)); u_o[2, 1] = 60                                  # 9th inning, behind by 1-2
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
    # 5 lead states (PENR rules): behind by 3+ / 1-2 / tied / ahead by 1-3 / ahead by 4+
    # top half: the home team fields, its lead = home - away
    assert B.PEN_LEAD["top"][at(2)] == 3 and B.PEN_LEAD["top"][at(-1)] == 1 and B.PEN_LEAD["top"][at(5)] == 4
    assert B.PEN_LEAD["top"][at(-3)] == 0
    # bottom half: the away team fields, its lead = away - home
    assert B.PEN_LEAD["bottom"][at(-1)] == 3 and B.PEN_LEAD["bottom"][at(1)] == 1 and B.PEN_LEAD["bottom"][at(0)] == 2


def test_bench_mixture_follows_the_penr_rules_and_defaults_stay():
    import bench_live as BL
    # PREREG_PENW / PREREG_PENR (H1 supported): 60 days, 5 lead states, KAPPA 0.25; the study's pick keeps 30 / 4 / 30
    assert (PU.BENCH_WINDOW_DAYS, PU.BENCH_STATES, PU.BENCH_KAPPA, PU.BENCH_TAU) == (60, 5, 0.25, 1.0)
    assert (PU.WINDOW_DAYS, PU.KAPPA, PU.TAU) == (30, 30.0, 20.0)
    t, d0 = RAW["late5"][len(RAW["late5"]) // 2][:2]
    day = date.fromisoformat(d0)
    exp = {}
    for tt, dd, p, ii, ls, n in RAW["late5"]:
        x = date.fromisoformat(dd)
        if tt == t and day - timedelta(days=60) <= x < day:
            exp.setdefault(p, np.zeros((3, 5)))[ii, ls] += n
    gd = {"datetime": {"officialDate": d0}, "game": {"type": "R"}}
    got = BL.pen_usage_for(gd, t, None)
    tired = PU.tired(t, day)
    assert set(got) == {p for p in exp if p not in tired}
    assert all(np.array_equal(got[p], exp[p]) for p in got)


def test_five_state_rows_collapse_to_the_four_state_rows():
    four = {}
    for t, d, p, ii, ls, n in RAW["late"]:
        four[(t, d, p, ii, ls)] = four.get((t, d, p, ii, ls), 0) + n
    five = {}
    for t, d, p, ii, ls, n in RAW["late5"]:
        k = (t, d, p, ii, {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}[ls])
        five[k] = five.get(k, 0) + n
    assert four == five
    assert [PU.lead_state5(x) for x in (-5, -3, -2, -1, 0, 1, 3, 4)] == [0, 0, 1, 1, 2, 3, 3, 4]


def test_bench_pool_weight_is_kappa_025():
    m, data, closer = _closer_case()
    av = {p: u for p, u in data.pen_avail_fld.items() if p != m.pitcher}
    W, tot = PU.weights(av, list(av), "B", states=5, tau=1.0)
    plan = BL.pen_plan(replace(m, diff=-2), data, {closer: "CLOSER"})
    ninth = next(r for r in plan if r["inning"] == 9)
    assert abs(ninth["other"] - (1 - sum(x for _, x in ninth["rows"]))) < 1e-12
    assert abs(sum(x for _, x in ninth["rows"]) - (1 - 0.25 / (0.25 + tot))) < 1e-9


def test_bench_later_halves_use_the_bench_kappa(monkeypatch):
    m, data, closer = _closer_case()
    b = BL._bench(True)
    out = {}
    for k in (0.25, 30.0):
        monkeypatch.setattr(PU, "BENCH_KAPPA", k)
        b._cache = {}
        out[k] = b.later_halves(m, data, m.lineup)[(9, "bottom")]
    assert np.abs(out[0.25] - out[30.0]).max() > 1e-4      # the pool weight in bench.py follows BENCH_KAPPA


def test_bench_later_halves_use_the_bench_tau(monkeypatch):
    m, data, closer = _closer_case()
    data.pen_avail_fld = {p: u + 0.5 for p, u in data.pen_avail_fld.items()}   # usage in every state, so TAU matters
    b = BL._bench(True)
    out = {}
    for t in (1.0, 20.0):
        monkeypatch.setattr(PU, "BENCH_TAU", t)
        b._cache = {}
        out[t] = b.later_halves(m, data, m.lineup)[(9, "bottom")]
    assert np.abs(out[1.0] - out[20.0]).max() > 1e-5      # bench.py passes BENCH_TAU to the mixture
