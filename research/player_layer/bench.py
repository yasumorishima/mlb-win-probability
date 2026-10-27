"""
Bench options on top of the player layer: for one game moment, the run
distribution of the rest of this half-inning and the win probability for
each option the bench has.

Options (offense unless noted):
  none                 play it straight
  pinch_hit(x)         x bats in the current batter's slot now and for the
                       rest of the game
  bunt                 the engine's observed bunt outcomes from this base-out
                       state (league table), then the next batter
  intentional_walk     defense: put the batter on (forced runners move)
  pitch_change(r)      defense: reliever r pitches the rest of this half
Every option is scored with the same later-half tables (game.py), so the
differences come from this half-inning. Handedness enters the
plate-appearance probabilities when `hands` is given (size_hand.py terms).

Gate (`check_gate`): with every multiplier = 1, `none` equals Engine.wp and
`bunt` / `intentional_walk` equal Engine.tactics(); see tests in
`selftest_bench.py`.

Not modelled: pinch runners (base running uses league transitions), the
on-deck batter being walked around, a reliever beyond this half-inning.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import game as G  # noqa: E402
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from wp_engine import END, MAX_RUNS, STATES, Engine, _force, _skey, _state  # noqa: E402

C = R.PA_CLASSES
RUNS = np.arange(MAX_RUNS + 1)
HAND_BETA, HAND_K = 0.5, 2000.0


@dataclass
class Moment:
    inning: int
    half: str                       # "top" / "bottom"
    outs: int
    bases: tuple                    # (1b, 2b, 3b) as 0/1
    diff: int                       # home - away
    sched: int
    lineup: list                    # batting team's 9 ids by absolute slot
    slot: int                       # current batter's slot
    pitcher: int
    fld_lineup: list                # fielding team's 9 ids by absolute slot
    fld_next: int                   # fielding team's next slot
    bat_pitcher: int                # pitcher the batting team will use next
    cur_is_starter: bool
    bat_pitcher_is_starter: bool
    park: np.ndarray | None = None  # class factors (7), 1.0 = average
    extra_runner: bool = True       # automatic runner in extra innings (not in the postseason)


@dataclass
class Data:
    """Player inputs at the moment (counts strictly before the game date)."""
    L: np.ndarray                                   # league class rates (7)
    bat: dict                                       # id -> class counts (7)
    pit: dict                                       # id -> class counts (7)
    pen_fld: np.ndarray                             # fielding team's bullpen counts (7)
    pen_bat: np.ndarray                             # batting team's bullpen counts (7)
    def_fld: np.ndarray                             # fielding team's in-play counts (4)
    def_bat: np.ndarray
    hands: dict | None = None                       # id -> (bat_side, pitch_hand)
    split_bat: dict = field(default_factory=dict)   # (id, pitcher hand) -> counts (7)
    split_pit: dict = field(default_factory=dict)   # (id, batter side) -> counts (7)
    platoon: dict = field(default_factory=dict)     # "LvR" etc -> league factor (7)
    prm: dict | None = None                         # shrinkage (layer.PARAMS when None)


def _counts(d: dict, k) -> np.ndarray:
    return d.get(k, np.zeros(len(C)))


def _platoon(data: Data, ids, pid) -> np.ndarray:
    out = np.ones((len(ids), len(C)))
    if not data.hands or pid is None:
        return out
    ph = data.hands.get(pid, (None, None))[1]
    if ph not in ("R", "L"):
        return out
    p_all = R.shrunk(_counts(data.pit, pid)[None, :], data.L, 200)[0]
    for k, b in enumerate(ids):
        sd = data.hands.get(b, (None, None))[0]
        bs = ("L" if ph == "R" else "R") if sd == "S" else sd
        F = data.platoon.get(f"{bs}v{ph}")
        if F is None:
            continue
        b_all = R.shrunk(_counts(data.bat, b)[None, :], data.L, 200)[0]
        sp = data.split_bat.get((b, ph), np.zeros(len(C)))
        pp = data.split_pit.get((pid, bs), np.zeros(len(C)))
        b_own = ((sp + HAND_K * b_all * F) / (sp.sum() + HAND_K)) / b_all / F
        p_own = ((pp + HAND_K * p_all * F) / (pp.sum() + HAND_K)) / p_all / F
        out[k] = F ** HAND_BETA * b_own * p_own
    return out


def _mult(data: Data, ids, pid, pcounts, dfn, park) -> np.ndarray:
    m = Ly.multipliers(np.vstack([_counts(data.bat, b) for b in ids]), pcounts[None, :], dfn[None, :], data.L,
                       prm=data.prm or Ly.PARAMS, park=park)
    return m * _platoon(data, ids, pid)


class Bench:
    def __init__(self, eng: Engine, ct: dict, unit: bool = False):
        self.eng, self.ct, self.unit = eng, ct, unit      # unit=True: every multiplier 1 (gate)

    def _dist(self, half, M):
        return Ly.run_dist(self.ct, half, np.ones_like(M) if self.unit else M)

    def later_halves(self, m: Moment, data: Data, lineup) -> dict:
        """H table for the later half-innings (v1 rules) with `lineup` for
        the batting team."""
        other = "bottom" if m.half == "top" else "top"
        X_cur = self._dist(m.half, _mult(data, lineup, m.pitcher, _counts(data.pit, m.pitcher), data.def_fld, m.park))
        X_pen = self._dist(m.half, _mult(data, lineup, None, data.pen_fld, data.def_fld, m.park))
        Y_bp = self._dist(other, _mult(data, m.fld_lineup, m.bat_pitcher, _counts(data.pit, m.bat_pitcher), data.def_bat, m.park))
        Y_pen = self._dist(other, _mult(data, m.fld_lineup, None, data.pen_bat, data.def_bat, m.park))
        first_y = (m.inning, "bottom") if m.half == "top" else (m.inning + 1, "top")

        def start_dist(i2, h2, st):
            if h2 == m.half:
                dd = X_cur if (m.cur_is_starter and i2 <= G.STARTER_LAST) else X_pen
                return dd[:, st].mean(axis=0)
            dd = Y_bp if (m.bat_pitcher_is_starter and i2 <= G.STARTER_LAST) else Y_pen
            return dd[m.fld_next, st] if (i2, h2) == first_y else dd[:, st].mean(axis=0)

        return G.game_wp(start_dist, m.sched, extra_runner=m.extra_runner)

    def _from(self, dist, slot, transitions) -> np.ndarray:
        """Runs in the rest of the half after immediate `transitions`
        [(next_state, runs, p)], the next batter in `slot`."""
        tot = np.zeros(MAX_RUNS + 1)
        for nxt, r, p in transitions:
            rest = np.eye(MAX_RUNS + 1)[0] if nxt == END else np.clip(dist[slot, STATES.index(nxt)], 0, None)
            rest = rest / rest.sum()
            sh = np.zeros(MAX_RUNS + 1)
            sh[r:] = rest[:MAX_RUNS + 1 - r]
            sh[-1] += rest[MAX_RUNS + 1 - r:].sum() if r else 0.0
            tot += p * sh
        return tot

    def options(self, m: Moment, data: Data, pinch_hitters=(), relievers=()) -> list[dict]:
        s = _state(m.outs, *m.bases)
        si = STATES.index(s)
        base_cur = _mult(data, m.lineup, m.pitcher, _counts(data.pit, m.pitcher), data.def_fld, m.park)
        dist = self._dist(m.half, base_cur)
        H = self.later_halves(m, data, m.lineup)
        res = []

        def add(name, side, runs, Hh):
            wp_home = G.wp_now(runs / runs.sum(), m.inning, m.half, m.diff, m.sched, Hh)
            off = wp_home if m.half == "bottom" else 1 - wp_home
            res.append({"option": name, "side": side, "runs_mean": float(RUNS @ runs / runs.sum()),
                        "p_score": float(1 - runs[0] / runs.sum()), "wp_offense": float(off)})

        add("none", "offense", np.clip(dist[m.slot, si], 0, None), H)
        nxt_slot = (m.slot + 1) % 9
        bt = self.eng.t.get("bunt", {}).get(_skey(s))
        if bt and m.outs < 2 and (m.bases[0] or m.bases[1]):
            add("bunt", "offense", self._from(dist, nxt_slot, [(tuple(n), r, p) for n, r, p in bt["dist"]]), H)
        if not all(m.bases):
            nb, r = _force(m.bases)
            add("intentional_walk", "defense", self._from(dist, nxt_slot, [(_state(m.outs, *nb), r, 1.0)]), H)
        for x in pinch_hitters:
            lu = list(m.lineup); lu[m.slot] = x
            d2 = self._dist(m.half, _mult(data, lu, m.pitcher, _counts(data.pit, m.pitcher), data.def_fld, m.park))
            add(f"pinch_hit:{x}", "offense", np.clip(d2[m.slot, si], 0, None), self.later_halves(m, data, lu))
        H_pen = self.later_halves(replace(m, cur_is_starter=False), data, m.lineup) if relievers else None
        for rp in relievers:                            # the starter is gone: later halves face the bullpen
            d2 = self._dist(m.half, _mult(data, m.lineup, rp, _counts(data.pit, rp), data.def_fld, m.park))
            add(f"pitch_change:{rp}", "defense", np.clip(d2[m.slot, si], 0, None), H_pen)
        none = res[0]
        for r in res:
            r["delta_wp_offense"] = r["wp_offense"] - none["wp_offense"]
            r["delta_runs"] = r["runs_mean"] - none["runs_mean"]
        return res
