"""
Whole-game player layer: home WP with team-specific run distributions in
every remaining half-inning.

* Current half: `layer.run_dist` for the batting lineup as known, the
  current pitcher, the defense and the park (exact lineup slot).
* Later halves of each team: the team's lineup as known now, against the
  pitcher expected for that inning: the defense's current pitcher if he
  is that team's starter, through inning STARTER_LAST; otherwise (and
  after that) the team's bullpen = all its relievers' earlier plate
  appearances this season, pooled. The fielding team's next turn at bat
  starts at its known next slot; every later start slot is averaged
  uniformly over the nine slots.
* Game recursion over (inning, half, score difference) with the engine's
  rules (walk-off, home half not played, automatic runner from inning
  sched + 1). With league distributions it reproduces `Engine._H` (gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import layer as Ly  # noqa: E402
from wp_engine import MAX_DIFF, MAX_INNING, MAX_RUNS, STATES  # noqa: E402

STARTER_LAST = 6
D = np.arange(-MAX_DIFF, MAX_DIFF + 1)          # index i <-> diff D[i] (home - away)
ND = len(D)
S_EMPTY = STATES.index((0, 0, 0, 0))
S_RUNNER2 = STATES.index((0, 0, 1, 0))


def _shift_index(sign: int, r: int) -> np.ndarray:
    """Index of clip(diff + sign * r) for every diff index."""
    return np.clip(D + sign * r, -MAX_DIFF, MAX_DIFF) + MAX_DIFF


SHIFT = {(sg, r): _shift_index(sg, r) for sg in (1, -1) for r in range(MAX_RUNS + 1)}


def game_wp(start_dist, sched: int, extra_runner: bool = True) -> dict:
    """Home WP at the start of every half-inning.

    start_dist(inning, half, start_state_index) -> P(runs = 0..MAX_RUNS) for that
    half. Returns H[(inning, half)] = array over diff indices, computed
    backwards from MAX_INNING like `Engine._H`."""
    H = {}

    def start_state(inn):
        return S_RUNNER2 if (extra_runner and inn > sched) else S_EMPTY

    def h(inn, half):
        if (inn, half) in H:
            return H[(inn, half)]
        if inn > MAX_INNING:
            v = 0.5 + 0.04 * np.sign(D)
            H[(inn, half)] = v
            return v
        dist = start_dist(inn, half, start_state(inn))
        H[(inn, half)] = after_half(inn, half, dist, sched, h)
        return H[(inn, half)]

    for inn in range(MAX_INNING + 1, 0, -1):
        h(inn, "bottom")
        h(inn, "top")
    return H


def after_half(inn, half, dist, sched, h) -> np.ndarray:
    """Home WP (over diff indices at the start of the half) given the
    half's run distribution; h(inn, half) gives later half starts."""
    total = np.zeros(ND)
    if half == "top":
        nxt = h(inn, "bottom")
        for r, p in enumerate(dist):
            if p == 0.0:
                continue
            di = SHIFT[(-1, r)]
            v = nxt[di].copy()
            if inn >= sched:
                v[D[di] > 0] = 1.0                    # home leads after the top of the last inning
            total += p * v
    else:
        nxt = h(inn + 1, "top")
        for r, p in enumerate(dist):
            if p == 0.0:
                continue
            di = SHIFT[(1, r)]
            if inn >= sched:
                d2 = D[di]
                v = np.where(d2 > 0, 1.0, np.where(d2 == 0, nxt[MAX_DIFF], 0.0))
            else:
                v = nxt[di]
            total += p * v
    return total


def wp_now(dist_now, inn, half, diff, sched, H) -> float:
    """Home WP before a PA, from the current half's remaining-run
    distribution and the half-start table H of the later halves."""
    if half == "bottom" and inn >= sched and diff > 0:
        return 1.0
    i = int(np.clip(diff, -MAX_DIFF, MAX_DIFF)) + MAX_DIFF
    h = lambda a, b: H[(a, b)] if (a, b) in H else np.full(ND, 0.5) + 0.04 * np.sign(D)
    v = after_half(inn, half, dist_now, sched, h)
    return float(v[i])


def team_half_dists(ct, half, lineup_M_by_pitcher: dict) -> dict:
    """Run distributions (9 slots x NS x runs) for one batting team against
    each pitcher option ('starter' / 'bullpen' / 'current')."""
    return {k: Ly.run_dist(ct, half, M) for k, M in lineup_M_by_pitcher.items()}
