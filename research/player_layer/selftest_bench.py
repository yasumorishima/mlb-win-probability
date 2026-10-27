"""
Gate for bench.py (no network, no 2026 data): with every multiplier = 1 the
options must reproduce the frozen engine: `none` = Engine.wp, `bunt` and
`intentional_walk` = Engine.tactics() (WP after the option, offense view).
Also checks that a better pinch hitter raises the offense WP and a better
reliever lowers it (direction only).

  python research/player_layer/selftest_bench.py --states <dir with pa_2021..2025>
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import bench as B  # noqa: E402
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from wp_engine import Engine  # noqa: E402

C = R.PA_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    a = ap.parse_args()
    eng = Engine.load(HERE.parents[1] / "engine_tables.json")
    pa = pd.concat([pd.read_parquet(a.states / f"pa_{s}.parquet") for s in range(2021, 2026)], ignore_index=True)
    ct = Ly.class_tables(eng, pa)
    pa["cls"] = pa.event_type.map(R.outcome_class)
    t = pa[pa.cls.isin(C) & pa.pa_complete]
    L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    z = np.zeros(len(C))
    sc = {"SB": {"intercept": 1.5, "x2": 0.0, "speed_z": 0.1, "missing": 0.0}}
    data = B.Data(L=L, bat={}, pit={}, pen_fld=z, pen_bat=z, def_fld=np.zeros(4), def_bat=np.zeros(4), speed_coef=sc)
    unit = B.Bench(eng, ct, unit=True)
    worst = 0.0
    n = 0
    for inn, half, outs, bases, diff in itertools.product((1, 5, 8, 9, 10), ("top", "bottom"), (0, 1, 2),
                                                          [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 0), (1, 0, 1)], (-2, -1, 0, 1)):
        if half == "bottom" and inn >= 9 and diff > 0:
            continue
        m = B.Moment(inn, half, outs, bases, diff, 9, list(range(9)), 0, 100, list(range(10, 19)), 0, 101, True, True)
        opts = {o["option"]: o for o in unit.options(m, data)}
        wp = eng.wp(inn, half, outs, bases, diff, 9)
        off = wp if half == "bottom" else 1 - wp
        worst = max(worst, abs(opts["none"]["wp_offense"] - off))
        steals = {k: v for k, v in opts.items() if k.startswith("steal_")}
        for t_ in eng.tactics(inn, half, outs, bases, diff, 9):
            if t_.tactic in ("steal_2b", "steal_3b"):
                o = [v for k, v in steals.items() if k.startswith(t_.tactic)][0]
                worst = max(worst, abs(o["breakeven"] - min(t_.breakeven, 1.0)))
            if t_.tactic == "sacrifice_bunt":
                worst = max(worst, abs(opts["bunt"]["wp_offense"] - t_.wp_if))
            if t_.tactic == "intentional_walk":
                worst = max(worst, abs(opts["intentional_walk"]["wp_offense"] - (1 - t_.wp_if)))
        n += 1
    print("gate states", n, "max |bench - engine|", worst)
    assert worst < 1e-6, worst
    # direction: a strong pinch hitter / reliever moves WP the expected way
    good_b = L * np.array([0.6, 1.6, 1.2, 1.3, 1, 1.8, 1]) * 2000
    weak_b = L * np.array([1.5, 0.6, 0.8, 0.7, 1, 0.4, 1]) * 2000
    data2 = B.Data(L=L, bat={i: weak_b for i in range(9)} | {50: good_b}, pit={60: L * np.array([1.6, .6, .8, .7, 1, .5, 1]) * 2000},
                   pen_fld=L * 3000, pen_bat=L * 3000, def_fld=np.zeros(4), def_bat=np.zeros(4))
    m = B.Moment(8, "bottom", 1, (0, 1, 0), 0, 9, list(range(9)), 3, 100, list(range(10, 19)), 0, 101, False, False)
    o = {x["option"]: x for x in B.Bench(eng, ct).options(m, data2, pinch_hitters=[50], relievers=[60])}
    assert o["pinch_hit:50"]["delta_wp_offense"] > 0, o
    assert o["pitch_change:60"]["delta_wp_offense"] < 0, o
    print({k: round(v["delta_wp_offense"], 4) for k, v in o.items()})
    # the walk sends up the NEXT batter: walking ahead of a strong hitter must
    # help the offense more than walking ahead of a weak one
    data3 = B.Data(L=L, bat={i: weak_b for i in range(9)}, pit={}, pen_fld=L * 3000, pen_bat=L * 3000,
                   def_fld=np.zeros(4), def_bat=np.zeros(4))
    data3.bat[3] = good_b                                  # current batter strong, next (slot 4) weak
    ibb_weak_next = {x["option"]: x for x in B.Bench(eng, ct).options(m, data3)}["intentional_walk"]["delta_wp_offense"]
    data3.bat[4] = good_b * 1.0; data3.bat[3] = weak_b     # current weak, next strong
    ibb_strong_next = {x["option"]: x for x in B.Bench(eng, ct).options(m, data3)}["intentional_walk"]["delta_wp_offense"]
    print("IBB delta: next weak", round(ibb_weak_next, 4), "next strong", round(ibb_strong_next, 4))
    assert ibb_strong_next > ibb_weak_next, (ibb_strong_next, ibb_weak_next)
    print("selftest_bench passed")


if __name__ == "__main__":
    main()
