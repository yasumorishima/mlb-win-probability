"""
game.py's recursion with run distributions that may depend on the score
difference at the start of the half (who pitches depends on the score:
a closer comes in with a small lead). game.py stays frozen; with
distributions that do not depend on the difference this returns exactly
game.game_wp / game.wp_now (gate: selftest at the bottom).

start_dist(inning, half, start_state) -> array (R,) or (ND, R): the runs
distribution of that half, for every start difference index (home - away).
"""

from __future__ import annotations

import numpy as np

import game as G
from game import D, MAX_DIFF, ND, SHIFT, S_EMPTY, S_RUNNER2
from wp_engine import MAX_INNING


def game_wp(start_dist, sched: int, extra_runner: bool = True) -> dict:
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
    dist = np.asarray(dist, float)
    if dist.ndim == 1:
        return G.after_half(inn, half, dist, sched, h)
    total = np.zeros(ND)
    if half == "top":
        nxt = h(inn, "bottom")
        for r in range(dist.shape[1]):
            p = dist[:, r]
            if not p.any():
                continue
            di = SHIFT[(-1, r)]
            v = nxt[di].copy()
            if inn >= sched:
                v[D[di] > 0] = 1.0
            total += p * v
    else:
        nxt = h(inn + 1, "top")
        for r in range(dist.shape[1]):
            p = dist[:, r]
            if not p.any():
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
    return G.wp_now(dist_now, inn, half, diff, sched, H)


if __name__ == "__main__":                              # gate: equals game.py for difference-free inputs
    rng = np.random.default_rng(1)
    R = 11
    tab = {}

    def sd(i, h, st):
        k = (i, h, st)
        if k not in tab:
            x = rng.random(R) ** 3
            tab[k] = x / x.sum()
        return tab[k]

    def sd_wide(i, h, st):
        return np.tile(sd(i, h, st), (ND, 1))
    for sched in (9, 7):
        H1, H2, H3 = G.game_wp(sd, sched), game_wp(sd, sched), game_wp(sd_wide, sched)
        err = max(max(abs(H1[k] - H2[k]).max(), abs(H1[k] - H3[k]).max()) for k in H1)
        print("sched", sched, "max diff", err)
        assert err < 1e-12
    print("gate OK")
