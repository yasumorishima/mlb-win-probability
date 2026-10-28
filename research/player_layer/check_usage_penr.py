"""
Gate: the bench's bullpen usage under the PENR rules (pen_usage.py: 60 days, 5 lead states,
KAPPA 0.25, TAU 1, from published/bullpen_usage_<season>.json) equals size_pen_x3.py's usage tables and
mixture weights on sampled team-days.

  python research/player_layer/check_usage_penr.py --states S --season 2026 --n 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import pa_rates as R  # noqa: E402
import pen_usage as PU  # noqa: E402
import size_pen_x3 as X  # noqa: E402
from size_game import game_context  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    X.WINDOW_DAYS, X.LEAD5, X.KAPPA, X.TAU = PU.BENCH_WINDOW_DAYS, True, PU.BENCH_KAPPA, PU.BENCH_TAU
    pa = game_context(R.lineup_slots(R.load_pa(a.states, [a.season])))
    tab = X.usage_tables(pa)
    keys = sorted(tab)
    rng = np.random.default_rng(20261028)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(a.n, len(keys)), replace=False)]
    bad_u = bad_w = n_rel = 0
    for team, dt in pick:
        want = tab[(team, dt)]
        got = PU.usage(int(team), pd.Timestamp(dt).date(), window=PU.BENCH_WINDOW_DAYS, states=PU.BENCH_STATES)
        if set(want) != set(got) or any(not np.array_equal(want[p], got[p]) for p in want):
            bad_u += 1
            continue
        av = set(list(want)[: max(1, len(want) - 2)])          # some relievers unavailable
        Wx, tx = X.mixture_weights(want, av, "B")
        Wp, tp = PU.weights(got, list(av), "B", states=PU.BENCH_STATES, tau=PU.BENCH_TAU)
        k7 = 7 - X.FIRST
        for (ii, ls), w in Wp.items():
            wx = Wx[(ii + k7, ls)] if ii < 3 else Wx[(10 - X.FIRST, ls)]
            if set(w) != set(wx) or any(abs(w[p] - wx[p]) > 1e-12 for p in w):
                bad_w += 1
        if abs(PU.pool_weight(tp, PU.BENCH_KAPPA) - X.KAPPA / (X.KAPPA + tx[k7])) > 1e-12:
            bad_w += 1
        n_rel += len(want)
    print(f"team-days {len(pick)}, relievers {n_rel}, usage mismatches {bad_u}, weight mismatches {bad_w}")
    sys.exit(1 if bad_u or bad_w else 0)


if __name__ == "__main__":
    main()
