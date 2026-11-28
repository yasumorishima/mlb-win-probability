"""
Faster bench options for the look-back view, without touching the frozen
player-layer code (`research/player_layer/{bench,layer,game_pen}.py`).

`FastBench` is the frozen `bench.Bench` with one change: `_dist` (the
rest-of-half run distribution `layer.run_dist`) is remembered across
moments of a game. The same lineup / pitcher / defense / park gives the
same multiplier matrix M, so most calls of the 「モデルなら動いた」 search
repeat an earlier one (2025-09-06 LAD@BAL: 3,154 calls, 472 distinct).

The key is the content of the inputs (half, the P and T tables, M, the
unit flag), never an object id: a dict's id can be reused by a new dict
after the old one is freed. Results are returned read-only, so a caller
that would change a remembered array in place fails loudly instead of
corrupting later answers.

`backend`: "python" = frozen `layer.run_dist`; "rust" = the `wpfast`
extension (same algorithm, see rust/wpfast); "auto" = rust if the
extension imports, else python. tests/test_fast_bench.py checks every
backend against the frozen Bench.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "research" / "player_layer"))
import bench as B  # noqa: E402  (frozen)
import layer as Ly  # noqa: E402  (frozen)
import game as G  # noqa: E402  (frozen)
import game_pen as GP  # noqa: E402  (frozen)
from wp_engine import MAX_INNING, MAX_RUNS  # noqa: E402

try:  # optional compiled backend
    import wpfast as _wpfast
except Exception:  # noqa: BLE001 - any import problem means "not available"
    _wpfast = None

CACHE_SIZE = 4096        # ~19 KB per entry (9 x 24 x 11 float64) -> <= ~80 MB


def rust_available() -> bool:
    return _wpfast is not None


def _digest(*arrays) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.digest()


def run_dist_rust(ct: dict, half: str, M: np.ndarray) -> np.ndarray:
    P, T = ct[half]["P"], ct[half]["T"]
    out = _wpfast.run_dist(np.ascontiguousarray(P, dtype=np.float64),
                           np.ascontiguousarray(T, dtype=np.float64),
                           np.ascontiguousarray(M, dtype=np.float64), Ly.MAX_RUNS)
    return np.asarray(out).reshape(9, Ly.NS, Ly.MAX_RUNS + 1)


class FastBench(B.Bench):
    def __init__(self, eng, ct, unit: bool = False, backend: str | None = None, cache_size: int = CACHE_SIZE):
        super().__init__(eng, ct, unit)
        backend = backend or os.environ.get("WP_BACKEND", "auto")
        if backend == "auto":
            backend = "rust" if rust_available() else "python"
        if backend == "rust" and not rust_available():
            raise RuntimeError("wpfast extension is not installed")
        self.backend = backend
        self._run = run_dist_rust if backend == "rust" else Ly.run_dist
        self._memo: OrderedDict = OrderedDict()
        self._avg_memo: dict = {}
        self._ct_digest: dict = {}           # id(table) -> (table, digest); holds the table so its id stays unique
        self.cache_size = cache_size
        self.stats = {"calls": 0, "hits": 0}

    def _table_digest(self, P, T) -> bytes:
        k = (id(P), id(T))
        got = self._ct_digest.get(k)
        if got is not None and got[0] is P and got[1] is T and not P.flags.writeable and not T.flags.writeable:
            return got[2]
        d = _digest(P, T)
        if not P.flags.writeable and not T.flags.writeable:     # only frozen tables are remembered by id
            self._ct_digest[k] = (P, T, d)
        return d

    def _dist(self, half, M, T=None):
        ct = self.ct if T is None else {**self.ct, half: {"P": self.ct[half]["P"], "T": T}}
        M = np.ones_like(M) if self.unit else M
        self.stats["calls"] += 1
        key = (half, self._table_digest(ct[half]["P"], ct[half]["T"]), _digest(np.asarray(M, dtype=np.float64)))
        got = self._memo.get(key)
        if got is not None:
            self.stats["hits"] += 1
            self._memo.move_to_end(key)
            return got
        out = np.asarray(self._run(ct, half, M))
        out.setflags(write=False)
        self._memo[key] = out
        if len(self._memo) > self.cache_size:
            self._memo.popitem(last=False)
        return out

    # --- copy of the frozen Bench._later_halves_pen (research/player_layer/bench.py);
    # only the last line differs: GP.game_wp -> self._game_wp (same H, see
    # tests/test_fast_bench.py, which also checks this copy against the frozen source)
    def _later_halves_pen(self, m: Moment, data: Data, lineup, exclude: tuple) -> dict:
        self._data = data
        other = "bottom" if m.half == "top" else "top"
        first_y = (m.inning, "bottom") if m.half == "top" else (m.inning + 1, "top")
        X_cur = self._rp_dist(m.half, lineup, m.pitcher, B._counts(data.pit, m.pitcher), data.def_fld, m.park)
        X_pen = self._rp_dist(m.half, lineup, "pool", data.pen_fld, data.def_fld, m.park)
        Y_bp = self._rp_dist(other, m.fld_lineup, m.bat_pitcher, B._counts(data.pit, m.bat_pitcher), data.def_bat, m.park)
        Y_pen = self._rp_dist(other, m.fld_lineup, "pool", data.pen_bat, data.def_bat, m.park)
        avg = self._avg                                   # memoised dd[:, st].mean(axis=0) (same values)
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

        stacks = {}

        def pen(tab, i2, h2, st):
            ii = 3 if i2 >= 10 else i2 - 7
            k = (id(tab), ii, h2, st)                     # tab lives through this call
            if k not in stacks:
                stacks[k] = np.stack([tab[(ii, ls, st)] for ls in B.PEN_LEAD[h2]])
            return stacks[k]

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

        return self._game_wp(start_dist, m.sched, extra_runner=m.extra_runner)

    def _avg(self, dd, st):
        """dd[:, st].mean(axis=0), remembered for the remembered (read-only) run tables."""
        if dd.flags.writeable:
            return dd[:, st].mean(axis=0)
        k = (id(dd), st)
        got = self._avg_memo.get(k)
        if got is None or got[0] is not dd:
            got = (dd, dd[:, st].mean(axis=0))            # holding dd keeps its id unique
            got[1].setflags(write=False)
            self._avg_memo[k] = got
            if len(self._avg_memo) > 4 * self.cache_size:
                self._avg_memo.clear()
        return got[1]

    def _game_wp(self, start_dist, sched: int, extra_runner: bool = True) -> dict:
        """GP.game_wp with every start distribution collected into one table
        first (the same calls game_wp makes), remembered by the table's
        content; the recursion runs in Rust when available."""
        tab = np.empty((MAX_INNING, 2, G.ND, MAX_RUNS + 1))
        for inn in range(1, MAX_INNING + 1):
            st = G.S_RUNNER2 if (extra_runner and inn > sched) else G.S_EMPTY
            for hi, half in enumerate(("top", "bottom")):
                tab[inn - 1, hi] = np.asarray(start_dist(inn, half, st), float)   # (R,) broadcasts over diffs
        if self.backend == "rust":                        # 0.1 ms: cheaper than hashing the table
            flat = np.asarray(_wpfast.game_wp(tab, int(sched), int(G.MAX_DIFF))).reshape(MAX_INNING + 1, 2, G.ND)
            return {(inn, half): flat[inn - 1, hi] for inn in range(1, MAX_INNING + 2) for hi, half in enumerate(("top", "bottom"))}
        key = ("H", sched, bool(extra_runner), _digest(tab))
        got = self._memo.get(key)
        if got is not None:
            self.stats["hits_h"] = self.stats.get("hits_h", 0) + 1
            self._memo.move_to_end(key)
            return dict(got)
        H = GP.game_wp(lambda i, h, s: tab[i - 1, 0 if h == "top" else 1], sched, extra_runner=extra_runner)
        for v in H.values():
            v.setflags(write=False)
        self._memo[key] = H
        if len(self._memo) > self.cache_size:
            self._memo.popitem(last=False)
        return dict(H)


class CheckedBench(FastBench):
    """FastBench that also runs the frozen Python for every run_dist and
    game_wp call and records the first call that differs by more than `tol`
    (checkpoint = stage + call number, fingerprint = where and how it
    differs). Used by tests/test_fast_bench.py and tools/check_wpfast.py."""

    def __init__(self, *a, tol: float = 1e-12, **k):
        super().__init__(*a, **k)
        self.tol = tol
        self.n = {"run_dist": 0, "game_wp": 0}
        self.max_diff = {"run_dist": 0.0, "game_wp": 0.0}
        self.first = None
        fast_run = self._run

        def run(ct, half, M):
            got, ref = np.asarray(fast_run(ct, half, M)), Ly.run_dist(ct, half, M)
            self._check("run_dist", got, ref, {"half": half, "index": "(slot, state, runs)"})
            return got
        self._run = run

    def _check(self, stage, got, ref, ctx):
        self.n[stage] += 1
        d = np.abs(got - ref)
        mx = float(d.max()) if d.size else 0.0
        self.max_diff[stage] = max(self.max_diff[stage], mx)
        if self.first is None and (got.shape != ref.shape or not mx <= self.tol):
            i = np.unravel_index(int(np.argmax(d)), d.shape)
            self.first = {"stage": stage, "call": self.n[stage], **ctx, "shape": list(ref.shape),
                          "max_abs": mx, "argmax": [int(x) for x in i], "n_over_tol": int((d > self.tol).sum()),
                          "of": int(d.size), "fast": float(got[i]), "frozen": float(ref[i]),
                          "sum_fast": float(got.sum()), "sum_frozen": float(ref.sum())}

    def _game_wp(self, start_dist, sched: int, extra_runner: bool = True) -> dict:
        H = super()._game_wp(start_dist, sched, extra_runner)
        R = GP.game_wp(start_dist, sched, extra_runner=extra_runner)   # the frozen recursion, same inputs
        keys = sorted(R)
        if sorted(H) != keys:
            self._check("game_wp", np.zeros(1), np.ones(1), {"note": "different (inning, half) keys"})
            return H
        self._check("game_wp", np.stack([H[k] for k in keys]), np.stack([R[k] for k in keys]),
                    {"sched": sched, "index": "(inning-half, diff)"})
        return H


def freeze_tables(ct: dict) -> dict:
    """Read-only copies of the class tables, so their digest can be remembered by id."""
    out = {}
    for h, v in ct.items():
        P, T = np.array(v["P"], dtype=np.float64), np.array(v["T"], dtype=np.float64)
        P.setflags(write=False)
        T.setflags(write=False)
        out[h] = {"P": P, "T": T}
    return out
