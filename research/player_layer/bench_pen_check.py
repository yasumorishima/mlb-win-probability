"""
Checks of the late-inning bullpen in the bench options (bench.py with
pen_avail_*), on recorded finished games (feeds not committed):

  python research/player_layer/bench_pen_check.py <feed.json> [...]

For every 7th-inning-or-later moment with a pinch-hit / pinch-run option:
  approx : |WP(pinch option) with the fast rule - with the exact mixture
           for that lineup| (points), max and mean;
  effect : how much the bullpen usage moves each option vs the pooled
           bullpen (points), and the spread of pitching-change options.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench_live as BL  # noqa: E402
import bench_replay as RP  # noqa: E402


def main() -> None:
    approx, moved, spread_pen, spread_pool = [], [], [], []
    for f in sys.argv[1:]:
        feed = json.load(open(f))
        tl = RP.timeline(feed)
        for t in tl:
            if t["inning"] < 7:
                continue
            got = RP.moment_at(feed, t["k"])
            if got is None:
                continue
            m, data, ex = got
            if data.pen_avail_fld is None:
                continue
            b = BL._bench(m.extra_runner)
            kw = dict(pinch_hitters=ex["bench"][:4], relievers=ex["bullpen"], pinch_runners=ex["bench"][:4])
            b.exact_lineups = False
            fast = {o["option"]: o["wp_offense"] for o in b.options(m, data, **kw)}
            b.exact_lineups = True
            exact = {o["option"]: o["wp_offense"] for o in b.options(m, data, **kw)}
            b.exact_lineups = False
            for k in fast:
                if k.startswith(("pinch_hit", "pinch_run")):
                    approx.append(100 * abs(fast[k] - exact[k]))
            fa, fb = data.pen_avail_fld, data.pen_avail_bat
            data.pen_avail_fld = data.pen_avail_bat = None
            pool = {o["option"]: o["wp_offense"] for o in b.options(m, data, **kw)}
            data.pen_avail_fld, data.pen_avail_bat = fa, fb
            moved += [100 * abs(exact[k] - pool[k]) for k in exact if k in pool]
            pc = [exact[k] for k in exact if k.startswith("pitch_change")]
            pp = [pool[k] for k in pool if k.startswith("pitch_change")]
            if len(pc) > 1:
                spread_pen.append(100 * (max(pc) - min(pc)))
                spread_pool.append(100 * (max(pp) - min(pp)))
            print(t["k"], t["inning"], t["top_bottom"], "n_avail", len(data.pen_avail_fld), len(data.pen_avail_bat), flush=True)
    a = np.array(approx)
    out = {"pinch_options": int(len(a)), "approx_max_pt": float(a.max()) if len(a) else None,
           "approx_mean_pt": float(a.mean()) if len(a) else None,
           "usage_vs_pool_mean_abs_pt": float(np.mean(moved)), "usage_vs_pool_max_pt": float(np.max(moved)),
           "pitch_change_spread_pt": {"usage": float(np.mean(spread_pen)), "pooled": float(np.mean(spread_pool))}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
