"""
Compare the fast look-back search with the frozen Python on recorded games.

  python tools/check_wpfast.py <feed.json> [<feed.json> ...] [--backend rust|python] [--tol 1e-12]

For every game: runs `missed_chances` + `game_review` three ways — the frozen
`bench.Bench`, and `fast_bench.CheckedBench` (the chosen backend, with every
run_dist / game_wp call recomputed by the frozen Python next to it). Prints
per game the seconds, the largest difference per stage and in the final rows,
and the first divergence above `tol` (stage, call number, where, how), if any.
Exit 1 when the rows differ in anything but floats or a float differs by more
than --row-tol (WP points).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_live as BL  # noqa: E402
import bench_replay as R  # noqa: E402
import fast_bench as FB  # noqa: E402
from wp_engine import Engine  # noqa: E402


def rows(feed, make):
    cache = {}
    old = BL._bench
    BL._bench = lambda er=True: cache.setdefault(er, make(er))
    try:
        t = time.perf_counter()
        out = {"missed": R.missed_chances(feed), "review": R.game_review(feed)}
        return json.loads(json.dumps(out, default=str)), time.perf_counter() - t, cache
    finally:
        BL._bench = old


def compare(a, b, path="", acc=None):
    acc = acc if acc is not None else {"max": 0.0, "where": "", "other": []}
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        for k in a:
            compare(a[k], b[k], f"{path}.{k}", acc)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > acc["max"]:
            acc["max"], acc["where"] = abs(a - b), path
    elif a != b:
        acc["other"].append(path)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feeds", nargs="+", type=Path)
    ap.add_argument("--backend", default="rust")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--row-tol", type=float, default=1e-9)
    a = ap.parse_args()
    eng = lambda er: Engine.load(ROOT / "engine_tables.json", extra_runner=er)  # noqa: E731
    bad = 0
    for f in a.feeds:
        feed = json.loads(f.read_text())
        ref, t0, _ = rows(feed, lambda er: FB.B.Bench(eng(er), BL._snapshot()["_ct"]))
        fast, t1, _ = rows(feed, lambda er: FB.FastBench(eng(er), FB.freeze_tables(BL._snapshot()["_ct"]), backend=a.backend))
        _, _, cc = rows(feed, lambda er: FB.CheckedBench(eng(er), FB.freeze_tables(BL._snapshot()["_ct"]),
                                                        backend=a.backend, tol=a.tol))
        acc = compare(ref, fast)
        chk = [b for b in cc.values()]
        out = {"feed": f.name, "frozen_s": round(t0, 2), f"{a.backend}_s": round(t1, 2),
               "rows_max_diff": acc["max"], "rows_max_at": acc["where"], "rows_other_diffs": acc["other"][:5],
               "calls": {k: sum(b.n[k] for b in chk) for k in ("run_dist", "game_wp")},
               "stage_max_diff": {k: max(b.max_diff[k] for b in chk) for k in ("run_dist", "game_wp")},
               "first_divergence": next((b.first for b in chk if b.first), None)}
        print(json.dumps(out))
        bad |= bool(acc["other"]) or acc["max"] > a.row_tol
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
