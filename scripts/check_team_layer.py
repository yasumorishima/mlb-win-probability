"""
Release gate for the team layer (research/wp_layer/PREREG.md, "What follows"):

1. On every 2025 game, d_prev and d_cur from team_layer.strengths (the MLB
   Stats API path the app uses) must match layer_features.team_strength (the
   training features) within 0.01 runs per game.
2. team_layer.team_wp, fed the training features, must reproduce the
   published 2026 TEAM predictions (research/wp_layer/published/pred_2026.parquet).

    python scripts/check_team_layer.py --states <dir with pa_YYYY.parquet>

Exit status 1 if either check fails. Prints the largest differences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "research/wp_layer"))
import layer_features as lf  # noqa: E402
import team_layer as tl  # noqa: E402
from wp_engine import Engine  # noqa: E402

TOL = 0.01


def check_strengths(states: Path) -> bool:
    pa = lf.load_pa(states, [2024, 2025])
    games = lf.game_table(pa)
    ref = lf.team_strength(games).join(games[["date", "season", "home_team", "away_team"]])
    ref = ref[ref.season == 2025]
    rows = []
    for pk, g in ref.iterrows():
        s = tl.strengths(int(g.home_team), int(g.away_team), str(g.date))
        if s is None:
            print("no strengths from the API for", pk, g.date)
            return False
        rows.append((pk, s["d_prev"] - g.d_prev, s["d_cur"] - g.d_cur))
    d = pd.DataFrame(rows, columns=["game_pk", "e_prev", "e_cur"]).set_index("game_pk")
    worst = d.abs().max()
    print(f"strengths: {len(d)} games of 2025, max |diff| d_prev {worst.e_prev:.2e} "
          f"d_cur {worst.e_cur:.2e} (tolerance {TOL})")
    bad = d[(d.e_prev.abs() > TOL) | (d.e_cur.abs() > TOL)]
    if len(bad):
        print(bad.head(20).to_string())
    return len(bad) == 0 and len(d) == ref.shape[0]


def check_formula(states: Path) -> bool:
    eng = Engine.load(APP / "engine_tables.json")
    d = lf.build(states, [2026], eng, league_exclude=[2026])
    pub = pd.read_parquet(APP / "research/wp_layer/published/pred_2026.parquet")
    assert (d.game_pk.to_numpy() == pub.game_pk.to_numpy()).all()
    assert (d.at_bat_index.to_numpy() == pub.at_bat_index.to_numpy()).all()
    p = np.array([tl.team_wp(pe, int(i), h, int(o), int(s), dp, dc) for pe, i, h, o, s, dp, dc in
                  zip(d.p_eng, d.inning, d.half, d.outs, d.scheduled_innings, d.d_prev, d.d_cur)])
    gap = float(np.max(np.abs(p - pub.p_TEAM.to_numpy())))
    print(f"formula: {len(p)} plate appearances of 2026, max |team_wp - published| {gap:.2e}")
    return gap < 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    a = ap.parse_args()
    ok_f = check_formula(a.states)
    ok_s = check_strengths(a.states)
    print("PASS" if ok_f and ok_s else "FAIL")
    sys.exit(0 if ok_f and ok_s else 1)


if __name__ == "__main__":
    main()
