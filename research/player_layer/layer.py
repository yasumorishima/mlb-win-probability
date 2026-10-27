"""
Stage B/C of the player layer: rest-of-half run distribution for the
actual lineup, current pitcher and defense, and the home WP from it.

State space: (base-out state, lineup slot 0..8). From (s, j) the PA ends
in class c with probability q_j(c | half, s) = P(OTHER | half, s) for the
non-PA class OTHER (slot unchanged), and otherwise
(1 - P(OTHER)) * L(c | half, s) * M_j(c) / normaliser, where M_j is the
batter-vs-pitcher-vs-defense multiplier (stage A, variant V5). Given the
class, the next state and runs follow the engine's `by_class` tables
(slot j -> j + 1). With every M_j = 1 this is exactly the engine's
league transition and must reproduce `Engine.runs` (gate).

WP = sum_r P(r more runs this half) * WP at the end of the half with r
runs (`Engine._wp_after(..., END, r, ...)`), i.e. later half-innings stay
league-average.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wp_engine import END, MAX_RUNS, STATES, Engine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pa_rates as R  # noqa: E402

C = R.PA_CLASSES
NB = [C.index(c) for c in R.NONBIP]
IB = [C.index(c) for c in R.BIP]
NS = len(STATES)                  # 24 live states
MAXR = 4                          # runs on one play are capped at 4 in the tables
PARAMS = {"kb1": 100, "kb2": 400, "kp": 200, "kpb": 2000, "kd": 4000}   # from size_pa (2022-2025)


def class_tables(eng: Engine, pa_train) -> dict:
    """Per half: P(class | state) incl. OTHER (from the training rows,
    the same rows the engine was fitted on) and per class the transition
    tensors T[c][r] (NS x (NS+1)), column NS = END."""
    from wp_engine import _occ, _state, outcome_class
    out = {}
    for half in ("top", "bottom"):
        sub = pa_train[pa_train.half == half]
        cnt = np.zeros((NS, len(C) + 1))
        for o, b1, b2, b3, e in sub[["outs", "on_1b", "on_2b", "on_3b", "event_type"]].itertuples(index=False):
            s = STATES.index(_state(o, _occ(b1), _occ(b2), _occ(b3)))
            k = outcome_class(e)
            cnt[s, C.index(k) if k in C else len(C)] += 1
        P = cnt / cnt.sum(axis=1, keepdims=True)
        T = np.zeros((len(C) + 1, MAXR + 1, NS, NS + 1))
        for ci, c in enumerate(C + ["OTHER"]):
            for si, s in enumerate(STATES):
                for nxt, r, p in eng.by_class.get((half, s, c), []):
                    T[ci, r, si, NS if nxt == END else STATES.index(nxt)] += p
        out[half] = {"P": P, "T": T}
    return out


def multipliers(B: np.ndarray, Pt: np.ndarray, D: np.ndarray, L: np.ndarray, prm=PARAMS,
                park: np.ndarray | None = None, alpha: float = 0.5) -> np.ndarray:
    """Stage-A multipliers (variant V5) for rows of batter counts B (n x 7),
    pitcher counts Pt (n x 7 or 1 x 7) and defense BIP counts D (n x 4);
    optional park class factors (7, 1.0 = average) applied as park ** alpha
    (alpha 0.5 from size_context.py)."""
    m = np.ones((len(B), len(C)))
    Lc = np.concatenate([L[NB], [L[IB].sum()]])
    Lb = L[IB] / L[IB].sum()
    for X, k1, k2 in ((B, prm["kb1"], prm["kb2"]), (np.broadcast_to(Pt, B.shape), prm["kp"], prm["kpb"])):
        Xc = np.column_stack([X[:, NB], X[:, IB].sum(axis=1)])
        mc = R.shrunk(Xc, Lc, k1) / Lc
        m[:, NB] *= mc[:, :3]
        m[:, IB] *= mc[:, 3:4]
        m[:, IB] *= R.shrunk(X[:, IB], Lb, k2) / Lb
    m[:, IB] *= R.shrunk(np.broadcast_to(D, (len(B), len(IB))), Lb, prm["kd"]) / Lb
    if park is not None:
        m = m * np.asarray(park)[None, :] ** alpha
    return m


def run_dist(ct: dict, half: str, M: np.ndarray) -> np.ndarray:
    """P(runs = 0..MAX_RUNS in the rest of the half) for every (state,
    slot): returns array (9, NS, MAX_RUNS + 1). M: (9, 7) multipliers by
    absolute lineup slot."""
    P, T = ct[half]["P"], ct[half]["T"]
    po = P[:, len(C)]                                    # P(OTHER | s)
    q = P[None, :, :len(C)] * M[:, None, :]              # (9, NS, 7)
    q = q / q.sum(axis=2, keepdims=True) * (1 - po)[None, :, None]
    n = 9 * NS
    Mr = np.zeros((MAXR + 1, n, n))
    A = np.zeros((MAXR + 1, n))                          # to END with r runs
    for j in range(9):
        rows = slice(j * NS, (j + 1) * NS)
        jn = (j + 1) % 9
        for ci in range(len(C)):
            W = q[j, :, ci][:, None] * T[ci]             # (MAXR+1, NS, NS+1) broadcast over r
            Mr[:, rows, jn * NS:(jn + 1) * NS] += W[:, :, :NS]
            A[:, rows] += W[:, :, NS]
        W = po[:, None] * T[len(C)]
        Mr[:, rows, rows] += W[:, :, :NS]
        A[:, rows] += W[:, :, NS]
    lu = lu_factor(np.eye(n) - Mr[0])
    out = np.zeros((MAX_RUNS + 1, n))
    for k in range(MAX_RUNS + 1):
        rhs = A[k].copy() if k <= MAXR else np.zeros(n)
        for r in range(1, MAXR + 1):
            if k - r >= 0:
                rhs += Mr[r] @ out[k - r]
        out[k] = lu_solve(lu, rhs)
    return out.T.reshape(9, NS, MAX_RUNS + 1)


def wp_from_runs(eng: Engine, dist: np.ndarray, inning: int, half: str, diff: int, sched: int) -> float:
    if half == "bottom" and inning >= sched and diff > 0:
        return 1.0
    return float(sum(p * eng._wp_after(inning, half, END, r, diff, sched) for r, p in enumerate(dist) if p > 0))
