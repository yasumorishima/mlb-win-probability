"""
Known-answer checks of analyze.py on synthetic data (no API, no real labels).

  python research/steal_claude/selftest.py

1. B knows the true probability, A and C are noisier: H2 supported, H1 not.
2. Raw files of the two arms swapped: analyze.py must refuse to run.
3. B is better than A only on the control period: the memory check fires
   and no skill is claimed for B.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EFFORT = "medium"


def make(work: Path, rng, b_good_test: bool, b_good_ctrl: bool) -> None:
    rows = []
    for split, n, games, start in (("test", 1800, 860, "2026-07-01"), ("control", 300, 250, "2026-04-01")):
        g = rng.integers(0, games, n) + (0 if split == "test" else 10**6)
        p_true = 1 / (1 + np.exp(-(1.5 + 0.8 * rng.standard_normal(n))))
        noise = 1 / (1 + np.exp(-(np.log(p_true / (1 - p_true)) + 0.9 * rng.standard_normal(n))))
        y = (rng.random(n) < p_true).astype(int)
        dates = pd.to_datetime(start) + pd.to_timedelta(rng.integers(0, 85, n), unit="D")
        for i in range(n):
            rows.append(dict(custom_id=f"{split}{i}", split=split, game_pk=int(g[i]), date=dates[i],
                             runner=1, pitcher=2, catcher=3, success=int(y[i]),
                             p_a=float(noise[i]), p_a_ctrl=float(noise[i]) if split == "control" else np.nan,
                             p_model=float(noise[i]), season_rate_before=0.8,
                             runner_sprint_prev=np.nan if i % 5 == 0 else 27.0,
                             catcher_pop_prev=1.95, runner_att_before=i % 9,
                             true_p=float(p_true[i])))
    df = pd.DataFrame(rows)
    df.drop(columns="true_p").to_parquet(work / "rows.parquet", index=False)
    raw = work / "raw"
    raw.mkdir()
    for arm in ("B", "C"):
        with open(raw / f"{arm}_test-control_{EFFORT}.jsonl", "w") as f:
            for r in df.itertuples():
                good = b_good_test if r.split == "test" else b_good_ctrl
                p = r.true_p if (arm == "B" and good) else r.p_model
                f.write(json.dumps({"custom_id": r.custom_id, "arm": arm, "effort": EFFORT,
                                    "model": "claude-opus-5-5", "stop_reason": "end_turn",
                                    "usage": {"input": 900, "output": 300},
                                    "text": json.dumps({"p_success": p, "reason": "x"})}) + "\n")


def run(work: Path):
    return subprocess.run([sys.executable, str(HERE / "analyze.py"), "--work", str(work), "--effort", EFFORT],
                          capture_output=True, text=True)


def main() -> None:
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as t:
        w = Path(t)
        make(w, rng, b_good_test=True, b_good_ctrl=True)
        p = run(w)
        assert p.returncode == 0, p.stderr
        r = json.loads((w / "results.json").read_text())
        assert r["supported"] == {"H1_C": False, "H2_B": True}, r["supported"]
        print("1 ok: H2 supported, H1 not", r["holm_adjusted_p"])

        (w / "raw" / f"B_test-control_{EFFORT}.jsonl").rename(w / "raw" / "tmp")
        (w / "raw" / f"C_test-control_{EFFORT}.jsonl").rename(w / "raw" / f"B_test-control_{EFFORT}.jsonl")
        (w / "raw" / "tmp").rename(w / "raw" / f"C_test-control_{EFFORT}.jsonl")
        p = run(w)
        assert p.returncode != 0 and "arm" in (p.stderr + p.stdout), p.stdout[-300:]
        print("2 ok: swapped arms refused")

    with tempfile.TemporaryDirectory() as t:
        w = Path(t)
        make(w, rng, b_good_test=False, b_good_ctrl=True)
        p = run(w)
        assert p.returncode == 0, p.stderr
        r = json.loads((w / "results.json").read_text())
        assert r["memory_check"]["B"]["consistent_with_memorization"] is True, r["memory_check"]
        assert r["skill_claimed"]["H2_B"] is False
        print("3 ok: control-only advantage flagged", r["memory_check"]["B"])


if __name__ == "__main__":
    main()
