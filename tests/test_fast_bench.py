"""fast_bench (look-back search speed): the frozen Python is the reference.

Every test compares FastBench (Python cache, and the Rust backend when the
wpfast extension is installed) with the frozen research/player_layer/bench.py
on the synthetic finished game. Tolerances were set after measuring on the
2025-09-06 LAD@BAL feed (cloud, 2026-11-28): Python cache 0 (bit-identical),
Rust run_dist <= 4.4e-16, game_wp 0, final rows <= 2.2e-14 WP points.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import bench_live as BL  # noqa: E402
import bench_replay as R  # noqa: E402
import fast_bench as FB  # noqa: E402
import game_fixture as F  # noqa: E402
from wp_engine import Engine  # noqa: E402

FEED = F.game_feed()
BACKENDS = ["python"] + (["rust"] if FB.rust_available() else [])
ROW_TOL = {"python": 0.0, "rust": 1e-9}


def _eng(er):
    return Engine.load(ROOT / "engine_tables.json", extra_runner=er)


def _rows(make):
    cache = {}
    old = BL._bench
    BL._bench = lambda er=True: cache.setdefault(er, make(er))
    try:
        out = {"missed": R.missed_chances(FEED, min_gain=-1e9, top=10_000), "review": R.game_review(FEED)}
        return json.loads(json.dumps(out, default=str)), cache
    finally:
        BL._bench = old


def _diff(a, b, acc):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _diff(a[k], b[k], acc)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _diff(x, y, acc)
    elif isinstance(a, float):
        acc[0] = max(acc[0], abs(a - b))
    else:
        assert a == b
    return acc[0]


@pytest.fixture(scope="module")
def frozen_rows():
    return _rows(lambda er: FB.B.Bench(_eng(er), BL._snapshot()["_ct"]))[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_fast_rows_equal_the_frozen_bench(frozen_rows, backend):
    fast, cache = _rows(lambda er: FB.FastBench(_eng(er), FB.freeze_tables(BL._snapshot()["_ct"]), backend=backend))
    assert frozen_rows["missed"], "the fixture game must give candidate moments"
    assert _diff(frozen_rows, fast, [0.0]) <= ROW_TOL[backend]
    st = next(iter(cache.values())).stats
    assert st["hits"] > 0 and st["hits"] < st["calls"]        # the memo is used, not everything is a hit


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_call_matches_the_frozen_python(backend):
    _, cache = _rows(lambda er: FB.CheckedBench(_eng(er), FB.freeze_tables(BL._snapshot()["_ct"]),
                                                backend=backend, tol=1e-12))
    for b in cache.values():
        assert b.n["run_dist"] > 0 and b.n["game_wp"] > 0
        assert b.first is None, b.first


def test_checkpoint_names_the_first_divergence():
    b = FB.CheckedBench(_eng(True), FB.freeze_tables(BL._snapshot()["_ct"]), backend="python", tol=1e-12)
    good = b._run
    b._run = lambda ct, half, M: good(ct, half, M)          # unchanged first call
    ct = b.ct
    M = np.ones((9, len(FB.Ly.C)))
    b._dist("top", M)
    assert b.first is None
    b._check("run_dist", np.array([[0.5, 0.5]]), np.array([[0.5, 0.6]]), {"half": "top"})
    f = b.first
    assert f["stage"] == "run_dist" and f["call"] == 2 and f["argmax"] == [0, 1] and f["n_over_tol"] == 1
    assert ct is b.ct


def test_copied_method_equals_the_frozen_source():
    """FastBench._later_halves_pen is the frozen method plus exactly these
    substitutions (memoised helpers with the same values, the game_wp call)."""
    import re
    ref = inspect.getsource(FB.B.Bench._later_halves_pen)
    ref = ref.replace("PEN_LEAD[", "B.PEN_LEAD[")
    ref = re.sub(r"(?<![\w.])_counts\(", "B._counts(", ref)
    subs = [
        ("        avg = lambda dd, st: dd[:, st].mean(axis=0)\n",
         "        avg = self._avg                                   # memoised dd[:, st].mean(axis=0) (same values)\n"),
        ("""        def pen(tab, i2, h2, st):
            ii = 3 if i2 >= 10 else i2 - 7
            return np.stack([tab[(ii, ls, st)] for ls in B.PEN_LEAD[h2]])
""", """        stacks = {}

        def pen(tab, i2, h2, st):
            ii = 3 if i2 >= 10 else i2 - 7
            k = (id(tab), ii, h2, st)                     # tab lives through this call
            if k not in stacks:
                stacks[k] = np.stack([tab[(ii, ls, st)] for ls in B.PEN_LEAD[h2]])
            return stacks[k]
"""),
        ("        return GP.game_wp(start_dist, m.sched, extra_runner=m.extra_runner)\n",
         "        return self._game_wp(start_dist, m.sched, extra_runner=m.extra_runner)\n"),
    ]
    for a, b in subs:
        assert ref.count(a) == 1, a
        ref = ref.replace(a, b)
    assert inspect.getsource(FB.FastBench._later_halves_pen) == ref


def test_key_covers_the_transition_tables():
    """A pinch runner changes T (not M): the memo must not return the old answer."""
    b = FB.FastBench(_eng(True), FB.freeze_tables(BL._snapshot()["_ct"]), backend="python")
    M = np.ones((9, len(FB.Ly.C)))
    a = b._dist("top", M)
    T2 = np.array(b.ct["top"]["T"])
    T2[0, 1] *= 1.3
    c = b._dist("top", M, T2)
    ref = FB.Ly.run_dist({**b.ct, "top": {"P": b.ct["top"]["P"], "T": T2}}, "top", M)
    assert not np.allclose(a, c) and np.array_equal(c, ref)


def test_remembered_arrays_are_read_only():
    b = FB.FastBench(_eng(True), FB.freeze_tables(BL._snapshot()["_ct"]), backend="python")
    a = b._dist("top", np.ones((9, len(FB.Ly.C))))
    with pytest.raises(ValueError):
        a[0, 0, 0] = 1.0


def test_app_uses_fastbench_and_falls_back_without_rust(monkeypatch):
    assert isinstance(BL._bench(True), FB.FastBench)
    monkeypatch.setattr(FB, "_wpfast", None)
    b = FB.FastBench(_eng(True), FB.freeze_tables(BL._snapshot()["_ct"]))
    assert b.backend == "python"
    with pytest.raises(RuntimeError):
        FB.FastBench(_eng(True), FB.freeze_tables(BL._snapshot()["_ct"]), backend="rust")
