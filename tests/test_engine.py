"""Tests for wp_engine on synthetic plate appearances with known rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wp_engine import Engine, fit_tables, _force, outcome_class  # noqa: E402

RNG = np.random.default_rng(7)
EVENTS = [("strikeout", 0.24), ("field_out", 0.44), ("walk", 0.09),
          ("single", 0.15), ("double", 0.05), ("home_run", 0.03)]


def _advance(bases, ev):
    b1, b2, b3 = bases
    if ev == "walk":
        (nb, runs) = _force(bases)
        return nb, runs
    if ev == "single":
        return (1, b1, b2), b3
    if ev == "double":
        return (0, 1, b1), b2 + b3
    if ev == "home_run":
        return (0, 0, 0), 1 + b1 + b2 + b3
    return bases, 0


def _simulate(n_halves=6000, half="top"):
    rows, pitches = [], []
    names, probs = zip(*EVENTS)
    gid = 0
    for h in range(n_halves):
        outs, bases = 0, (0, 0, 0)
        ab = 0
        while outs < 3:
            ev = RNG.choice(names, p=probs)
            if ev in ("strikeout", "field_out"):
                nb, runs, no = bases, 0, outs + 1
            else:
                nb, runs = _advance(bases, ev)
                no = outs
            rows.append(dict(game_pk=gid, at_bat_index=ab, half=half, outs=outs,
                             on_1b=bases[0] or None, on_2b=bases[1] or None, on_3b=bases[2] or None,
                             outs_after=no, post_1b=nb[0] or None, post_2b=nb[1] or None,
                             post_3b=nb[2] or None, runs_on_play=runs, event_type=ev,
                             is_bunt=False))
            # counts: every PA passes 0-0; strikeouts tend to reach 0-2, walks 3-0
            pitches.append(dict(game_pk=gid, at_bat_index=ab, balls=0, strikes=0))
            if ev == "strikeout" or RNG.random() < 0.2:
                pitches.append(dict(game_pk=gid, at_bat_index=ab, balls=0, strikes=2))
            if ev == "walk" or RNG.random() < 0.05:
                pitches.append(dict(game_pk=gid, at_bat_index=ab, balls=3, strikes=0))
            outs, bases = no, nb
            ab += 1
        gid += 1
    return rows, pitches


@pytest.fixture(scope="module")
def engine():
    top, ptop = _simulate(half="top")
    bot, pbot = _simulate(half="bottom")
    for r in bot:
        r["game_pk"] += 10**6
    for r in pbot:
        r["game_pk"] += 10**6
    pa = pd.DataFrame(top + bot)
    pitch = pd.DataFrame(ptop + pbot)
    return Engine(fit_tables(pa, pitch, min_bunts=1))


def test_symmetric_world_starts_near_even(engine):
    # both halves come from the same process: home edge only from batting last
    wp = engine.wp(1, "top", 0, (0, 0, 0), 0)
    assert 0.47 < wp < 0.56


def test_wp_in_unit_interval_and_monotone_in_lead(engine):
    for inning in (1, 5, 9, 11):
        for half in ("top", "bottom"):
            vals = [engine.wp(inning, half, 1, (1, 0, 0), d) for d in range(-4, 5)
                    if not (half == "bottom" and inning >= 9 and d > 0)]
            assert all(0.0 <= v <= 1.0 for v in vals)
            assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_walkoff_rules(engine):
    # home already ahead in the bottom of the ninth: game over
    assert engine.wp(9, "bottom", 0, (0, 0, 0), 1) == 1.0
    # home ahead after the top of the ninth with three outs pending is near 1
    assert engine.wp(9, "top", 2, (0, 0, 0), 3) > 0.97
    # tied, bottom 9th, bases loaded, nobody out: very likely win
    assert engine.wp(9, "bottom", 0, (1, 1, 1), 0) > 0.85


def test_extra_inning_runner_matters(engine):
    with_runner = Engine(engine.t, extra_runner=True).wp(10, "top", 0, (0, 1, 0), 0)
    assert 0.3 < with_runner < 0.7


def test_count_layer_moves_the_right_way(engine):
    args = (7, "bottom", 1, (1, 0, 0), 0)
    behind = engine.wp_at_count(*args, 0, 2)
    ahead = engine.wp_at_count(*args, 3, 0)
    assert ahead > behind
    # 0-0 is reached by every PA: close to the count-free value
    assert abs(engine.wp_at_count(*args, 0, 0) - engine.wp(*args)) < 0.01


def test_steal_breakeven_is_a_probability(engine):
    res = {t.tactic: t for t in engine.tactics(8, "bottom", 0, (1, 0, 0), 0)}
    be = res["steal_2b"].breakeven
    assert be is not None and 0.4 < be < 0.95
    ibb = res["intentional_walk"]
    assert ibb.side == "defense" and ibb.delta < 0


def test_force():
    assert _force((0, 0, 0)) == ((1, 0, 0), 0)
    assert _force((0, 1, 0)) == ((1, 1, 0), 0)
    assert _force((1, 0, 1)) == ((1, 1, 1), 0)
    assert _force((1, 1, 1)) == ((1, 1, 1), 1)


def test_outcome_classes():
    assert outcome_class("strikeout") == "K"
    assert outcome_class("intent_walk") == "BB"
    assert outcome_class("grounded_into_double_play") == "OUT"
    assert outcome_class("caught_stealing_2b") == "OTHER"
