"""
Checks run before the first API call (PREREG.md), no network, no key:

1. analyze.py on synthetic data with known answers: an arm that knows the
   true probability must be supported, an arm that is A plus noise must not,
   a failed answer must be filled with A and counted, and a record of the
   wrong arm must stop the analysis.
2. build_table's causal records on the real tables (training dates only):
   batter / pitcher counts recomputed by brute force for random rows.
3. Bunt labels by event type on 2024-2025 (sacrifice = success, strikeout
   and double play = failure).

  python research/bunt_ibb_claude/selftest.py --states <dir with pa_*.parquet>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import build_table as bt  # noqa: E402


def _synthetic(work: Path, wrong_arm: bool = False) -> dict:
    """Writes the synthetic work dir; returns the expected test log loss of
    arm B per part (failed answers filled with A)."""
    expect = {}
    rng = np.random.default_rng(1)
    (work / "raw").mkdir(parents=True)
    recs = {"B": [], "C": []}
    for part, n_test, base in (("bunt", 600, 0.8), ("obp", 3000, 0.33)):
        n = n_test + 300
        logit = np.log(base / (1 - base)) + rng.normal(0, 0.8, n)
        p_true = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p_true).astype(int)
        p_a = np.full(n, base)                      # A knows only the base rate
        split = np.array(["test"] * n_test + ["control"] * 300)
        dates = np.where(split == "test", "2026-08-15", "2026-05-15")
        df = pd.DataFrame({"custom_id": [f"{part}-{i}-1" for i in range(n)], "part": part, "split": split,
                           "game_pk": np.arange(n) // 3, "date": dates, "batter": 1, "pitcher": 2, "y": y,
                           "p_a": p_a, "p_a_ctrl": np.where(split == "control", base + 0.05, np.nan),
                           "p_model": np.where(split == "control", base + 0.05, p_a), "lg_rate": base, "bat_n_before": rng.integers(0, 300, n)})
        df.to_parquet(work / f"rows_{part}.parquet", index=False)
        p_b = p_true.copy()
        if part == "obp":
            p_b[0] = p_a[0]
        t = split == "test"
        expect[part] = float(analyze.ll(p_b[t], y[t].astype(float)).mean())
        c = ~t                                          # control: A is the refit A_ctrl
        expect[part + "_A_control"] = float(analyze.ll(np.full(c.sum(), base + 0.05), y[c].astype(float)).mean())
        p_c = np.clip(p_a + rng.normal(0, 0.05, n), 0.01, 0.99)
        for i, cid in enumerate(df.custom_id):
            txt_b = json.dumps({"p_success": float(p_true[i]), "reason": "x"})
            if part == "obp" and i == 0:
                txt_b = "not json"                      # must be filled with A
            recs["B"].append({"custom_id": cid, "arm": "B", "model": "m", "stop_reason": "end_turn",
                              "usage": {"input": 1, "output": 1}, "text": txt_b})
            recs["C"].append({"custom_id": cid, "arm": "C", "model": "m", "stop_reason": "end_turn",
                              "usage": {"input": 1, "output": 1},
                              "text": json.dumps({"p_success": float(p_c[i]), "reason": "x"})})
    if wrong_arm:
        recs["C"][5]["arm"] = "B"
    for arm in ("B", "C"):
        with open(work / "raw" / f"{arm}_test-control_medium.jsonl", "w") as f:
            for r in recs[arm]:
                f.write(json.dumps(r) + "\n")
    return expect


def check_analysis() -> None:
    with tempfile.TemporaryDirectory() as d:
        w = Path(d)
        expect = _synthetic(w)
        res = analyze.analyze(w, "medium")
        for key, v in expect.items():
            part = key.split("_")[0]
            got = (res["by_part"][part]["control"]["A"] if key.endswith("_A_control")
                   else res["by_part"][part]["test"]["B"])["log_loss"]
            assert abs(got - v) < 1e-12, (key, got, v)
        assert res["supported"]["H2_bunt_B"] and res["supported"]["H2_obp_B"], res["holm_adjusted_p"]
        assert not res["supported"]["H1_bunt_C"] and not res["supported"]["H1_obp_C"], res["holm_adjusted_p"]
        assert res["n"]["B_obp_test_filled_with_A"] == 1, res["n"]
        assert res["n"]["B_bunt_test_filled_with_A"] == 0, res["n"]
        assert res["skill_claimed"]["H2_obp_B"]
    with tempfile.TemporaryDirectory() as d:
        w = Path(d)
        _synthetic(w, wrong_arm=True)
        r = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--work", str(w), "--effort", "medium"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "expected" in (r.stderr + r.stdout), r.stderr
    print("analysis on synthetic data: ok")


def check_records(states: Path) -> None:
    pa = bt.load_pa(states)
    hist = bt.pa_obp_label(pa)
    u = bt.obp_units(pa)
    u = u[(u.date >= bt.TRAIN_START) & (u.date < bt.CTRL_START)]
    u = bt.add_records(u, hist, "batter", "ob", "bat")
    u = bt.add_records(u, hist, "pitcher", "ob", "pit")
    rng = np.random.default_rng(7)
    for i in rng.choice(len(u), 40, replace=False):
        r = u.iloc[i]
        hb = hist[(hist.batter == r.batter) & (hist.date < r.date)]
        hp = hist[(hist.pitcher == r.pitcher) & (hist.date < r.date)]
        assert (r.bat_n_before, r.bat_s_before) == (len(hb), int(hb.ob.sum())), (r.custom_id if "custom_id" in r else i)
        assert (r.pit_n_before, r.pit_s_before) == (len(hp), int(hp.ob.sum()))
    b = bt.bunt_units(pa)
    b = b[(b.date >= bt.TRAIN_START) & (b.date < bt.CTRL_START)]
    b = bt.add_records(b, b, "batter", "y", "bat")
    for i in rng.choice(len(b), 40, replace=False):
        r = b.iloc[i]
        hb = b[(b.batter == r.batter) & (b.date < r.date)]
        assert (r.bat_n_before, r.bat_s_before) == (len(hb), int(hb.y.sum()))
    print("causal records (80 random training rows): ok")

    lab = bt.bunt_units(pa)
    lab = lab[(lab.date >= "2024-01-01") & (lab.date < "2026-01-01")]
    m = lab.groupby("event_type").y.agg(["size", "mean"])
    assert m.loc["sac_bunt", "mean"] == 1.0, m
    for e in ("strikeout", "double_play", "grounded_into_double_play"):
        assert m.loc[e, "mean"] == 0.0, m
    print("bunt labels by event type (2024-2025): ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    a = ap.parse_args()
    check_analysis()
    check_records(a.states)
    print("selftest passed")


if __name__ == "__main__":
    main()
