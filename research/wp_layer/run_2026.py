"""
Frozen evaluation of the correction layers on 2026 (see PREREG.md).

Fits every arm on 2022-2025 and scores 2026 once. Writes (with --dry: dry/,
fitted on 2022-2024 and scored on 2025, a training season)
  published/results_2026.json   metrics, bootstrap intervals, verdicts
  published/pred_2026.parquet   per plate appearance: y and each arm's prediction
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer_features as lf  # noqa: E402
from size_layer import ARMS, design  # noqa: E402

APP = HERE.parents[1]
sys.path.insert(0, str(APP))
from wp_engine import Engine  # noqa: E402

STATES = Path.home() / "claude-scratch/mlbwp/data/states"
TRAIN = [2022, 2023, 2024, 2025]
TEST = 2026
B = 2000
SEED = 20261015
N_TRAIN_ROWS = 733260          # from the sizing run
N_TEST_ROWS = 184327           # reports/engine_check_2026.json
OUT = HERE / "published"
# md5 of the frozen inputs, taken when PREREG.md was frozen (checked in the live run)
FROZEN_MD5 = {
    "wp_engine.py": "12972ac8bb23a2dc8c152c07cbfd7a56",
    "engine_tables.json": "ffd4b264f96c29434ba21afb4ac7c9fb",
    "pa_2021.parquet": "10947f7add3b8821ddaa70705b845058",
    "pa_2022.parquet": "bbacd8f9703d5922d615903ebd693ca3",
    "pa_2023.parquet": "c8e4f89027179b1b09465a3d73580011",
    "pa_2024.parquet": "7d7f79769fae15f0c431d6389b4624ea",
    "pa_2025.parquet": "9d82f22f73ddd0189714b5c277a315ef",
    "pa_2026.parquet": "1c54a8d674846aac475aa87ce27fcc3c",
}


def input_md5() -> dict:
    paths = {"wp_engine.py": APP / "wp_engine.py", "engine_tables.json": APP / "engine_tables.json",
             **{f"pa_{s}.parquet": STATES / f"pa_{s}.parquet" for s in range(2021, 2027)}}
    return {k: hashlib.md5(p.read_bytes()).hexdigest() for k, p in paths.items()}


def fit(c, cols, y):
    X = np.column_stack([c[k] for k in cols])
    m, s = X.mean(0), X.std(0)
    lr = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8).fit((X - m) / s, y)
    pred = lambda cc: lr.predict_proba((np.column_stack([cc[k] for k in cols]) - m) / s)[:, 1]
    coef = dict(zip(cols, (lr.coef_[0] / s).tolist()))
    return pred, coef


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib(p, y):
    bins = np.clip((p * 10).astype(int), 0, 9)
    return [{"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": int((bins == b).sum()),
             "mean_pred": float(p[bins == b].mean()), "observed": float(y[bins == b].mean())}
            for b in range(10) if (bins == b).any()]


def main():
    if not __debug__:
        raise SystemExit("run without -O: the checks are asserts")
    global TRAIN, TEST, OUT
    dry = "--dry" in sys.argv
    if dry:   # machinery check on training seasons only: 2022-2024 -> 2025
        TRAIN, TEST, OUT = [2022, 2023, 2024], 2025, HERE / "dry"
    assert lf.K_TEAM == 30 and lf.K_SP == 200
    md5_in = input_md5()
    if not dry:
        bad = {k: v for k, v in md5_in.items() if v != FROZEN_MD5[k]}
        assert not bad, f"frozen inputs changed: {bad}"
    eng = Engine.load(APP / "engine_tables.json")
    d = lf.build(STATES, TRAIN + [TEST], eng, league_exclude=[TEST])
    tr, te = d[d.season.isin(TRAIN)], d[d.season == TEST]
    if not dry:
        assert len(tr) == N_TRAIN_ROWS, len(tr)
        assert len(te) == N_TEST_ROWS, len(te)
    assert not set(tr.game_pk) & set(te.game_pk)
    for c in ("d_prev", "d_cur", "sp"):
        assert te[c].notna().all() and tr[c].notna().all(), c
    assert te.has_prev.all() and tr.has_prev.all()

    mu = float(tr.runs_on_play.sum() / tr.groupby(["game_pk", "inning", "half"]).ngroups)
    ctr, cte = design(tr, mu), design(te, mu)
    ytr, y = tr.home_won.to_numpy(float), te.home_won.to_numpy(float)

    preds = {"ENGINE": te.p_eng.to_numpy()}
    coefs = {}
    for arm, cols in ARMS.items():
        f, coefs[arm] = fit(ctr, cols, ytr)
        preds[arm] = f(cte)

    import platform
    import sklearn
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__}
    res = {"train": TRAIN, "test": TEST, "versions": versions, "n_test": int(len(te)),
           "n_games": int(te.game_pk.nunique()), "mu": mu, "coef": coefs, "arms": {}}
    band = (preds["ENGINE"] >= 0.3) & (preds["ENGINE"] < 0.5)
    phase = np.digitize(te.r.to_numpy(), [1 / 3, 2 / 3])        # 0 late, 1 mid, 2 early
    for arm, p in preds.items():
        own = (p >= 0.3) & (p < 0.5)
        res["arms"][arm] = {
            "brier": brier(p, y), "log_loss": logloss(p, y),
            "gap35_engine_band": float(p[band].mean() - y[band].mean()),
            "gap35_own_band": float(p[own].mean() - y[own].mean()),
            "brier_by_phase": {k: brier(p[phase == i], y[phase == i])
                               for i, k in enumerate(("late", "mid", "early"))},
            "calibration": calib(p, y)}

    # game-cluster bootstrap
    g_idx = pd.factorize(te.game_pk)[0]
    G = g_idx.max() + 1
    cnt = np.bincount(g_idx, minlength=G).astype(float)
    sse = {a: np.bincount(g_idx, (p - y) ** 2, minlength=G) for a, p in preds.items()}
    bn = np.bincount(g_idx, band.astype(float), minlength=G)
    by = np.bincount(g_idx, y * band, minlength=G)
    bp = {a: np.bincount(g_idx, p * band, minlength=G) for a, p in preds.items()}
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(float)
    n_b = W @ cnt
    br = {a: (W @ s) / n_b for a, s in sse.items()}
    gap = {a: (W @ bp[a] - W @ by) / (W @ bn) for a in preds}

    def lower(diff_reps):          # H: diff < 0
        return {"point": None, "ci95": [float(np.percentile(diff_reps, 2.5)),
                                        float(np.percentile(diff_reps, 97.5))],
                "p_one_sided": float((1 + (diff_reps >= 0).sum()) / (B + 1))}

    tests = {
        "H1_TEAM_minus_RECAL_brier": (br["TEAM"] - br["RECAL"],
                                      res["arms"]["TEAM"]["brier"] - res["arms"]["RECAL"]["brier"]),
        "H2_TEAM_minus_ENGINE_brier": (br["TEAM"] - br["ENGINE"],
                                       res["arms"]["TEAM"]["brier"] - res["arms"]["ENGINE"]["brier"]),
        "H3_absgapTEAM_minus_absgapENGINE": (
            np.abs(gap["TEAM"]) - np.abs(gap["ENGINE"]),
            abs(res["arms"]["TEAM"]["gap35_engine_band"]) - abs(res["arms"]["ENGINE"]["gap35_engine_band"])),
    }
    out_tests = {}
    for k, (reps, point) in tests.items():
        t = lower(reps)
        t["point"] = float(point)
        out_tests[k] = t
    # Holm over H1 and H2, one-sided alpha 0.05; H3 is reported with its interval only
    out_tests["H3_absgapTEAM_minus_absgapENGINE"]["SUPPORTED"] = None
    order = sorted([k for k in out_tests if not k.startswith("H3")],
                   key=lambda k: out_tests[k]["p_one_sided"])
    alive = True
    for i, k in enumerate(order):
        thr = 0.05 / (2 - i)
        ok = alive and out_tests[k]["p_one_sided"] <= thr
        out_tests[k]["holm_threshold"] = thr
        out_tests[k]["SUPPORTED"] = bool(ok)
        alive = ok
    res["tests"] = out_tests

    report = {}
    for a, b_ in (("RECAL", "ENGINE"), ("SP", "RECAL"), ("INGAME", "RECAL"),
                  ("FULL", "TEAM"), ("FULL", "RECAL")):
        reps = br[a] - br[b_]
        report[f"{a}_minus_{b_}_brier"] = {
            "point": res["arms"][a]["brier"] - res["arms"][b_]["brier"],
            "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}
    res["report_only"] = report
    res["script_md5"] = {f: hashlib.md5((HERE / f).read_bytes()).hexdigest()
                         for f in ("run_2026.py", "layer_features.py", "size_layer.py")}
    res["input_md5"] = md5_in

    OUT.mkdir(exist_ok=True)
    (OUT / f"results_{TEST}.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"game_pk": te.game_pk.to_numpy(), "at_bat_index": te.at_bat_index.to_numpy(),
                  "y": y, **{f"p_{a}": p for a, p in preds.items()}}).to_parquet(
        OUT / f"pred_{TEST}.parquet", index=False)
    for a in preds:
        r = res["arms"][a]
        print(f"{a:7s} brier {r['brier']:.5f} logloss {r['log_loss']:.5f} "
              f"gap35(eng band) {r['gap35_engine_band']:+.4f}")
    for k, t in out_tests.items():
        print(k, round(t["point"], 5), [round(x, 5) for x in t["ci95"]],
              "p", round(t["p_one_sided"], 4),
              {True: "SUPPORTED", False: "NOT SUPPORTED", None: "(reported only)"}[t["SUPPORTED"]])


if __name__ == "__main__":
    main()
