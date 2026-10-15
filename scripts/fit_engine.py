"""
Fit the engine tables and check them on a held-out season.

  python scripts/fit_engine.py --states data/states --train 2021 2022 2023 2024 2025 \
      --test 2026 --out engine_tables.json --report reports/engine_check.json

Every plate appearance of the training seasons is used, including the
plays where a half ended on a baserunning out (they are how some halves
end). Transitions are base-out level, so seven-inning games are kept.
The held-out check scores the
home win probability before every plate appearance of the test season
against the final result (Brier score, log loss, calibration by decile)
and compares it with the old normal-approximation engine
(scripts/old_engine.py, the hand-tuned function the app used before).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wp_engine import Engine, fit_tables, _occ


def load(states: Path, kind: str, seasons: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(states / f"{kind}_{s}.parquet") for s in seasons],
                     ignore_index=True)


def score(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    out = {"n": int(len(y)),
           "brier": float(np.mean((p - y) ** 2)),
           "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    bins = np.clip((p * 10).astype(int), 0, 9)
    out["calibration"] = [
        {"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": int((bins == b).sum()),
         "mean_pred": float(p[bins == b].mean()) if (bins == b).any() else None,
         "observed": float(y[bins == b].mean()) if (bins == b).any() else None}
        for b in range(10)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--test", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    pa = load(a.states, "pa", a.train)
    pitch = load(a.states, "pitch", a.train)
    tables = fit_tables(pa, pitch)
    eng = Engine(tables)

    # league-average |WP swing| per PA on the training seasons (LI denominator)
    sample = pa.sample(n=min(20000, len(pa)), random_state=0)
    swings = [eng.leverage(r.inning, r.half, r.outs, (_occ(r.on_1b), _occ(r.on_2b), _occ(r.on_3b)),
                           r.home_score - r.away_score, r.scheduled_innings)
              for r in sample.itertuples(index=False)]
    tables["league_swing"] = float(np.mean(swings))
    tables["train_seasons"] = a.train
    a.out.write_text(json.dumps(tables))

    test = load(a.states, "pa", a.test)
    p_new = np.array([
        eng.wp(r.inning, r.half, r.outs, (_occ(r.on_1b), _occ(r.on_2b), _occ(r.on_3b)),
               r.home_score - r.away_score, r.scheduled_innings)
        for r in test.itertuples(index=False)])
    y = test.home_won.to_numpy().astype(float)

    report = {"train": a.train, "test": a.test, "engine": score(p_new, y)}
    try:
        from old_engine import calculate_wp  # scripts/old_engine.py
        p_old = np.array([
            calculate_wp(r.inning, r.half, r.outs, (_occ(r.on_1b), _occ(r.on_2b), _occ(r.on_3b)),
                         r.home_score - r.away_score)
            for r in test.itertuples(index=False)])
        report["old_engine"] = score(p_old, y)
    except Exception as e:  # report the failure instead of hiding it
        report["old_engine_error"] = repr(e)
    report["home_rate"] = float(y.mean())
    report["constant_home_rate"] = score(np.full_like(y, pa.home_won.mean()), y)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1))
    for k in ("engine", "old_engine", "constant_home_rate"):
        if k in report:
            print(k, {m: round(report[k][m], 5) for m in ("brier", "log_loss")}, "n", report[k]["n"])


if __name__ == "__main__":
    main()
