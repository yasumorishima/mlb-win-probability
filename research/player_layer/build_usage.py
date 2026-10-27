"""
Relief usage tables for the bench screen (pen_usage.py):

  python research/player_layer/build_usage.py --states S --season 2026

writes published/bullpen_usage_<season>.json:
  late   : [team, date, pitcher, inning - 7, lead state, relief PAs] for innings 7-9
  relief : [team, date, pitcher] for every relief appearance
(the same quantities size_pen.py computes from the states tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import pa_rates as R  # noqa: E402
from pen_usage import PUB, lead_state  # noqa: E402
from size_game import game_context  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    a = ap.parse_args()
    pa = game_context(R.lineup_slots(R.load_pa(a.states, [a.season])))
    first = pa.groupby(["game_pk", "half", "inning"], sort=False)[["home_score", "away_score"]].first()
    r = pa[pa.is_relief & pa.inning.isin((7, 8, 9))].join(
        first.rename(columns={"home_score": "h0", "away_score": "a0"}), on=["game_pk", "half", "inning"])
    lead = (r.h0 - r.a0).where(r.half == "top", r.a0 - r.h0)
    r = r.assign(ls=[lead_state(int(x)) for x in lead], ii=r.inning - 7)
    late = r.groupby(["fld_team", "date", "pitcher", "ii", "ls"]).size().reset_index()
    rel = pa[pa.is_relief][["fld_team", "date", "pitcher"]].drop_duplicates()
    out = {"season": a.season,
           "late": [[int(t), d.date().isoformat(), int(p), int(i), int(l), int(n)] for t, d, p, i, l, n in late.itertuples(index=False)],
           "relief": [[int(t), d.date().isoformat(), int(p)] for t, d, p in rel.itertuples(index=False)]}
    path = PUB / f"bullpen_usage_{a.season}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(path, len(out["late"]), "late rows", len(out["relief"]), "relief rows", path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
