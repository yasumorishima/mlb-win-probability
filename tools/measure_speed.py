"""
Time the engine pieces and the look-back calls once, on this machine.

  OMP_NUM_THREADS=1 python tools/measure_speed.py --states <states dir> --feed <finished feed.json>

Prints one JSON object: seconds per call for fit_tables (PA loop and the
count layer separately), Engine construction (_runs_dist), first and repeated
wp() queries, and the look-back calls on one recorded game (game_review,
missed_chances). Reads no outcome for any study; changes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import wp_engine as W  # noqa: E402


def t(f, *a, n=1, **k):
    best = None
    for _ in range(n):
        s = time.perf_counter()
        out = f(*a, **k)
        d = time.perf_counter() - s
        best = d if best is None else min(best, d)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--feed", type=Path)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args()
    res = {}
    pa = pd.concat([pd.read_parquet(a.states / f"pa_{s}.parquet") for s in a.seasons], ignore_index=True)
    pitch = pd.concat([pd.read_parquet(a.states / f"pitch_{s}.parquet") for s in a.seasons], ignore_index=True)
    res["rows"] = {"pa": len(pa), "pitch": len(pitch)}
    res["fit_tables_pa_only_s"], tab0 = t(W.fit_tables, pa)
    res["class_given_count_s"], _ = t(W._class_given_count, pa, pitch)
    res["fit_tables_full_s"], tab = t(W.fit_tables, pa, pitch)
    ref = json.loads((ROOT / "engine_tables.json").read_text())
    tab = json.loads(json.dumps(tab))
    res["fit_equals_engine_tables"] = all(tab[k] == ref[k] for k in tab)  # ref adds train_seasons / league_swing
    res["engine_init_s"], eng = t(W.Engine, ref, n=3)
    res["runs_dist_one_half_s"], _ = t(eng._runs_dist, "top", n=3)
    e2 = W.Engine(ref)
    res["wp_first_query_s"], _ = t(e2.wp, 1, "top", 0, (0, 0, 0), 0)
    res["wp_repeat_query_s"], _ = t(e2.wp, 7, "bottom", 1, (1, 0, 1), -1, n=50)
    if a.feed:
        import bench_replay as R
        feed = json.loads(a.feed.read_text())
        res["game_review_s"], rows = t(R.game_review, feed)
        res["game_review_calls"] = len(rows)
        res["missed_chances_s"], _ = t(R.missed_chances, feed)
        res["timeline_len"] = len(R.timeline(feed))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
