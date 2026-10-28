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
import game_pen as GP  # noqa: E402
import layer as Ly  # noqa: E402
import pen_usage as PU  # noqa: E402
import pa_rates as R  # noqa: E402
from wp_engine import END, MAX_RUNS, STATES, Engine, _force, _skey, _state  # noqa: E402

C = R.PA_CLASSES
RUNS = np.arange(MAX_RUNS + 1)
# lead state of the fielding team for every start difference (home - away): home fields in the top half
PEN_LEAD = {"top": np.array([PU.lead_state_n(int(d), PU.BENCH_STATES) for d in G.D]),
            "bottom": np.array([PU.lead_state_n(int(-d), PU.BENCH_STATES) for d in G.D])}
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
    runner_ids: tuple = (None, None, None)   # who is on 1st / 2nd / 3rd


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
    speed: dict = field(default_factory=dict)       # id -> sprint speed z (previous season)
    speed_coef: dict = field(default_factory=dict)  # size_speed.py: E1 / E2 / E3 / SB coefficients
    # late-inning bullpen by usage (size_pen.py variant B): available reliever -> usage (3 innings x 4 lead
    # states) over the 30 days before the game; None = the pooled bullpen in every later half (v1)
    pen_avail_fld: dict | None = None
    pen_avail_bat: dict | None = None


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


def _speed_T(T: np.ndarray, base: int, dz: float, coef: dict) -> np.ndarray:
    """Per-class transitions with the runner on `base` (1 or 2) dz standard
    deviations faster than league average: the odds of the extra base
    (size_speed.py E1 / E2 / E3) are multiplied by exp(beta * dz)."""
    T = T.copy()
    fams = []
    if base == 2:     # E1: single, runner on 2nd, 3rd empty -> he scores
        fams.append((C.index("1B"), lambda st: st[2] == 1 and st[3] == 0, lambda nxt, r: r >= 1, coef["E1"]["speed_z"]))
    if base == 1:     # E2: single, runner on 1st, 2nd/3rd empty -> 3rd or home; E3: double -> home
        fams.append((C.index("1B"), lambda st: st[1] == 1 and st[2] == 0 and st[3] == 0,
                     lambda nxt, r: r >= 1 or (nxt != END and nxt[3] == 1), coef["E2"]["speed_z"]))
        fams.append((C.index("2B"), lambda st: st[1] == 1 and st[2] == 0 and st[3] == 0,
                     lambda nxt, r: r >= 1, coef["E3"]["speed_z"]))
    for ci, in_fam, extra, beta in fams:
        for si, st in enumerate(STATES):
            if not in_fam(st):
                continue
            ext = np.zeros((T.shape[1], T.shape[3]), dtype=bool)
            for r in range(T.shape[1]):
                for j in range(T.shape[3]):
                    nxt = END if j == len(STATES) else STATES[j]
                    ext[r, j] = extra(nxt, r)
            row = T[ci, :, si, :]
            pe = row[ext].sum()
            tot = row.sum()
            if tot <= 0 or pe <= 0 or pe >= tot:
                continue
            q = pe / tot
            q2 = 1 / (1 + (1 - q) / q * np.exp(-beta * dz))
            row = np.where(ext, row * (q2 / q), row * ((1 - q2) / (1 - q)))
            T[ci, :, si, :] = row
    return T


class Bench:
    def __init__(self, eng: Engine, ct: dict, unit: bool = False):
        self.eng, self.ct, self.unit = eng, ct, unit      # unit=True: every multiplier 1 (gate)
        self._cache = {}
        self.exact_lineups = False                        # True: bullpen mixture recomputed for every lineup (slow)

    def _rp_dist(self, half, lineup, pid, pcounts, dfn, park):
        k = (half, tuple(lineup), pid, id(pcounts) if pid == "pool" else None)
        if k not in self._cache:
            self._cache[k] = self._dist(half, _mult(self._data, lineup, None if pid == "pool" else pid, pcounts, dfn, park))
        return self._cache[k]

    def _late_tables(self, m, data, half, lineup, avail: dict, pool_counts, dfn, slot_pick) -> dict:
        """{(inning idx, lead state, start state): runs (R,)} for innings 7+ of
        the team pitching in `half` (size_pen.py variant B)."""
        W, tot = PU.weights(avail, list(avail), "B", states=PU.BENCH_STATES)
        pool_w = PU.pool_weight(tot, PU.BENCH_KAPPA)
        pool = self._rp_dist(half, lineup, "pool", pool_counts, dfn, m.park)
        out = {}
        for (ii, ls), w in W.items():
            sw = sum(w.values())
            for st in (G.S_EMPTY, G.S_RUNNER2):
                base = slot_pick(pool, st)
                if sw <= 0:
                    out[(ii, ls, st)] = base
                    continue
                mix = sum(x * slot_pick(self._rp_dist(half, lineup, p, _counts(data.pit, p), dfn, m.park), st)
                          for p, x in w.items()) / sw
                out[(ii, ls, st)] = pool_w * base + (1 - pool_w) * mix
        return out

    def _dist(self, half, M, T=None):
        ct = self.ct if T is None else {**self.ct, half: {"P": self.ct[half]["P"], "T": T}}
        return Ly.run_dist(ct, half, np.ones_like(M) if self.unit else M)

    def later_halves(self, m: Moment, data: Data, lineup, exclude=()) -> dict:
        """H table for the later half-innings (v1 rules) with `lineup` for
        the batting team. With bullpen usage in `data` (pen_avail_*), innings
        7+ face the usage mixture of the available relievers (size_pen.py B),
        without the current pitcher and `exclude` for the fielding team.
        For a lineup other than m.lineup the fielding team's mixture is
        the pooled bullpen plus the mixture's difference from the pool
        measured with m.lineup (approximation; error in bench_pen_check.py)."""
        if data.pen_avail_fld is not None or data.pen_avail_bat is not None:
            return self._later_halves_pen(m, data, lineup, tuple(exclude))
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

    def _later_halves_pen(self, m: Moment, data: Data, lineup, exclude: tuple) -> dict:
        self._data = data
        other = "bottom" if m.half == "top" else "top"
        first_y = (m.inning, "bottom") if m.half == "top" else (m.inning + 1, "top")
        X_cur = self._rp_dist(m.half, lineup, m.pitcher, _counts(data.pit, m.pitcher), data.def_fld, m.park)
        X_pen = self._rp_dist(m.half, lineup, "pool", data.pen_fld, data.def_fld, m.park)
        Y_bp = self._rp_dist(other, m.fld_lineup, m.bat_pitcher, _counts(data.pit, m.bat_pitcher), data.def_bat, m.park)
        Y_pen = self._rp_dist(other, m.fld_lineup, "pool", data.pen_bat, data.def_bat, m.park)
        avg = lambda dd, st: dd[:, st].mean(axis=0)
        nxt = lambda dd, st: dd[m.fld_next, st]
        drop = {m.pitcher, *exclude}
        av_f = {p: u for p, u in (data.pen_avail_fld or {}).items() if p not in drop}
        av_b = {p: u for p, u in (data.pen_avail_bat or {}).items() if p != m.bat_pitcher}
        exact = self.exact_lineups or tuple(lineup) == tuple(m.lineup)
        key = ("Xl", exclude, tuple(lineup) if exact else tuple(m.lineup))
        if key not in self._cache:
            self._cache[key] = self._late_tables(m, data, m.half, list(key[2]), av_f, data.pen_fld, data.def_fld, avg)
        Xl = self._cache[key]
        if not exact:                                        # pool(lineup) + (mixture - pool) at m.lineup
            p0 = self._rp_dist(m.half, m.lineup, "pool", data.pen_fld, data.def_fld, m.park)
            Xl = {k: v + avg(X_pen, k[2]) - avg(p0, k[2]) for k, v in Xl.items()}
        if ("Yl",) not in self._cache:
            self._cache[("Yl",)] = (self._late_tables(m, data, other, m.fld_lineup, av_b, data.pen_bat, data.def_bat, avg),
                                    self._late_tables(m, data, other, m.fld_lineup, av_b, data.pen_bat, data.def_bat, nxt))
        Yl_avg, Yl_first = self._cache[("Yl",)]

        def pen(tab, i2, h2, st):
            ii = 3 if i2 >= 10 else i2 - 7
            return np.stack([tab[(ii, ls, st)] for ls in PEN_LEAD[h2]])

        def start_dist(i2, h2, st):
            if h2 == m.half:
                if m.cur_is_starter and i2 <= G.STARTER_LAST:
                    return avg(X_cur, st)
                return avg(X_pen, st) if i2 < 7 or data.pen_avail_fld is None else pen(Xl, i2, h2, st)
            if m.bat_pitcher_is_starter and i2 <= G.STARTER_LAST:
                return nxt(Y_bp, st) if (i2, h2) == first_y else avg(Y_bp, st)
            if i2 < 7 or data.pen_avail_bat is None:
                return nxt(Y_pen, st) if (i2, h2) == first_y else avg(Y_pen, st)
            return pen(Yl_first if (i2, h2) == first_y else Yl_avg, i2, h2, st)

        return GP.game_wp(start_dist, m.sched, extra_runner=m.extra_runner)

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

    def options(self, m: Moment, data: Data, pinch_hitters=(), relievers=(), pinch_runners=()) -> list[dict]:
        self._cache = {}
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
        # steal: success / failure with the same batter up; the runner's chance from size_speed.py (SB)
        sb = data.speed_coef.get("SB")
        for frm, to in ((1, 2), (2, 3)):
            if not m.bases[frm - 1] or m.bases[to - 1] or not sb:
                continue
            b_ok = list(m.bases); b_ok[frm - 1] = 0; b_ok[to - 1] = 1
            b_out = list(m.bases); b_out[frm - 1] = 0
            ok = self._from(dist, m.slot, [(_state(m.outs, *b_ok), 0, 1.0)])
            out = self._from(dist, m.slot, [(_state(m.outs + 1, *b_out), 0, 1.0)])
            w_ok = G.wp_now(ok / ok.sum(), m.inning, m.half, m.diff, m.sched, H)
            w_out = G.wp_now(out / out.sum(), m.inning, m.half, m.diff, m.sched, H)
            w_now = G.wp_now(np.clip(dist[m.slot, si], 0, None) / np.clip(dist[m.slot, si], 0, None).sum(),
                             m.inning, m.half, m.diff, m.sched, H)
            if m.half == "top":
                w_ok, w_out, w_now = 1 - w_ok, 1 - w_out, 1 - w_now
            rid = m.runner_ids[frm - 1]
            z = data.speed.get(rid)
            zz = sb["intercept"] + sb["x2"] * (to == 3) + (sb["speed_z"] * z if z is not None else sb["missing"])
            p = 1 / (1 + np.exp(-zz))
            be = (w_now - w_out) / (w_ok - w_out) if w_ok > w_out else 1.0
            mix = p * ok / ok.sum() + (1 - p) * out / out.sum()
            add(f"steal_{to}b:{rid}", "offense", mix, H)
            res[-1].update({"p_success": float(p), "breakeven": float(min(max(be, 0.0), 1.0)), "go": bool(p > be)})
        # pinch runner: faster runner on the base whose extra bases are modelled (2nd with 3rd empty, else
        # 1st with 2nd / 3rd empty); he also takes the runner's place in the lineup
        if pinch_runners and data.speed_coef.get("E1"):
            base = 2 if (m.bases[1] and not m.bases[2]) else (1 if (m.bases[0] and not m.bases[1] and not m.bases[2]) else None)
            rid = m.runner_ids[base - 1] if base else None
            if base and rid is not None:
                z0 = data.speed.get(rid, 0.0) or 0.0
                T0 = _speed_T(self.ct[m.half]["T"], base, z0, data.speed_coef)
                w0 = None
                for x in pinch_runners:
                    zx = data.speed.get(x)
                    if zx is None or zx <= z0:
                        continue
                    lu = list(m.lineup)
                    if rid in lu:
                        lu[lu.index(rid)] = x
                    Tx = _speed_T(self.ct[m.half]["T"], base, zx, data.speed_coef)
                    Mx = _mult(data, lu, m.pitcher, _counts(data.pit, m.pitcher), data.def_fld, m.park)
                    if w0 is None:
                        d0 = self._dist(m.half, base_cur, T0)
                        w0 = G.wp_now(np.clip(d0[m.slot, si], 0, None) / np.clip(d0[m.slot, si], 0, None).sum(),
                                      m.inning, m.half, m.diff, m.sched, H)
                    dx = self._dist(m.half, Mx, Tx)
                    Hx = self.later_halves(m, data, lu)
                    wx = G.wp_now(np.clip(dx[m.slot, si], 0, None) / np.clip(dx[m.slot, si], 0, None).sum(),
                                  m.inning, m.half, m.diff, m.sched, Hx)
                    if m.half == "top":
                        dw = (1 - wx) - (1 - w0)
                    else:
                        dw = wx - w0
                    runs = np.clip(dx[m.slot, si], 0, None)
                    res.append({"option": f"pinch_run:{x}", "side": "offense", "runs_mean": float(RUNS @ runs / runs.sum()),
                                "p_score": float(1 - runs[0] / runs.sum()), "wp_offense": res[0]["wp_offense"] + float(dw),
                                "speed_gain_sd": float(zx - z0), "replaces": rid})
        usage = data.pen_avail_fld is not None
        H_pen = self.later_halves(replace(m, cur_is_starter=False), data, m.lineup) if relievers and not usage else None
        for rp in relievers:                            # the starter is gone: later halves face the bullpen
            d2 = self._dist(m.half, _mult(data, m.lineup, rp, _counts(data.pit, rp), data.def_fld, m.park))
            # with usage: rp is not available later (using the closer now costs the 9th)
            Hr = self.later_halves(replace(m, cur_is_starter=False), data, m.lineup, exclude=(rp,)) if usage else H_pen
            add(f"pitch_change:{rp}", "defense", np.clip(d2[m.slot, si], 0, None), Hr)
        none = res[0]
        for r in res:
            r["delta_wp_offense"] = r["wp_offense"] - none["wp_offense"]
            r["delta_runs"] = r["runs_mean"] - none["runs_mean"]
        return res
