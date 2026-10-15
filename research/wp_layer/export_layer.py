"""
Export the TEAM layer of the frozen run for the app (see RESULTS.md).

results_2026.json has the nine coefficients but not the intercept. This
redoes the TEAM fit on 2022-2025 exactly as run_2026.py does, checks that it
reproduces the recorded coefficients, takes the intercept in raw units, and
checks that the raw-unit formula gives the published 2026 predictions.
Writes published/layer_team.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer_features as lf  # noqa: E402
import run_2026 as R  # noqa: E402
from size_layer import ARMS, design  # noqa: E402
from wp_engine import Engine  # noqa: E402

import pandas as pd  # noqa: E402


def main():
    res = json.loads((HERE / "published/results_2026.json").read_text())
    pub = pd.read_parquet(HERE / "published/pred_2026.parquet")
    assert R.input_md5() == R.FROZEN_MD5
    eng = Engine.load(R.APP / "engine_tables.json")
    d = lf.build(R.STATES, R.TRAIN + [R.TEST], eng, league_exclude=[R.TEST])
    tr, te = d[d.season.isin(R.TRAIN)], d[d.season == R.TEST]
    mu = float(tr.runs_on_play.sum() / tr.groupby(["game_pk", "inning", "half"]).ngroups)
    cols = ARMS["TEAM"]
    ctr, cte = design(tr, mu), design(te, mu)
    X = np.column_stack([ctr[k] for k in cols])
    m, s = X.mean(0), X.std(0)
    lr = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8).fit(
        (X - m) / s, tr.home_won.to_numpy(float))
    coef = lr.coef_[0] / s
    intercept = float(lr.intercept_[0] - (lr.coef_[0] * m / s).sum())
    rec = np.array([res["coef"]["TEAM"][k] for k in cols])
    assert np.max(np.abs(coef - rec)) < 1e-9, np.max(np.abs(coef - rec))

    Xte = np.column_stack([cte[k] for k in cols])
    p_raw = 1 / (1 + np.exp(-(intercept + Xte @ coef)))
    assert (te.game_pk.to_numpy() == pub.game_pk.to_numpy()).all()
    assert (te.at_bat_index.to_numpy() == pub.at_bat_index.to_numpy()).all()
    gap = float(np.max(np.abs(p_raw - pub.p_TEAM.to_numpy())))
    assert gap < 1e-9, gap

    out = {
        "source": "research/wp_layer run_2026.py (frozen 0cc376d), TEAM arm, fitted on 2022-2025",
        "terms": cols,
        "intercept": intercept,
        "coef": dict(zip(cols, coef.tolist())),
        "K_TEAM": lf.K_TEAM,
        "logit_clip": 1e-4,
        "definitions": {
            "L": "logit of the engine's home WP, p clipped to [1e-4, 1-1e-4]",
            "r": "share of regulation outs still to be played, floored at one half-inning: "
                 "max(1 - done/(6*sched), 3/(6*sched)), done = ((inning-1)*2 + bottom)*3 + outs",
            "d_prev": "home minus away run differential per game, previous regular season",
            "d_cur": "home minus away run differential per game this season on earlier dates, "
                     "each team's mean times n/(n+K_TEAM)",
        },
        "check": {"max_abs_coef_diff": float(np.max(np.abs(coef - rec))),
                  "max_abs_pred_diff_2026": gap},
    }
    (HERE / "published/layer_team.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["check"]), "intercept", intercept)


if __name__ == "__main__":
    main()
