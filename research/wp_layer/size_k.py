"""Training seasons only: K_TEAM sweep, TEAM coefficients, gain by game phase."""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_features as lf
from size_layer import design, ARMS, STATES, SEASONS, APP
from wp_engine import Engine

eng = Engine.load(APP / "engine_tables.json")
for K in (10, 30, 60, 120):
    lf.K_TEAM = K
    d = lf.build(STATES, SEASONS, eng)
    tot = {"RECAL": [], "TEAM": []}
    phase = {"RECAL": np.zeros(3), "TEAM": np.zeros(3)}
    nph = np.zeros(3)
    for hold in SEASONS:
        tr, te = d[d.season != hold], d[d.season == hold]
        mu = float(tr.runs_on_play.sum() / tr.groupby(["game_pk", "inning", "half"]).ngroups)
        ctr, cte = design(tr, mu), design(te, mu)
        ytr, yte = tr.home_won.to_numpy(float), te.home_won.to_numpy(float)
        ph = np.digitize(te.r.to_numpy(), [1/3, 2/3])
        for arm in tot:
            cols = ARMS[arm]
            X = np.column_stack([ctr[k] for k in cols]); m, s = X.mean(0), X.std(0)
            lr = LogisticRegression(C=1e6, max_iter=5000, tol=1e-8).fit((X - m) / s, ytr)
            p = lr.predict_proba((np.column_stack([cte[k] for k in cols]) - m) / s)[:, 1]
            se = (p - yte) ** 2
            tot[arm].append(se.mean())
            for k in range(3):
                phase[arm][k] += se[ph == k].sum()
            if arm == "TEAM" and K == 30 and hold == 2025:
                print("coef (raw units)", dict(zip(cols, np.round(lr.coef_[0] / s, 4))))
        for k in range(3):
            nph[k] += (ph == k).sum()
    print(f"K={K}: RECAL {np.mean(tot['RECAL']):.5f} TEAM {np.mean(tot['TEAM']):.5f}  "
          "TEAM-RECAL by phase (late, mid, early):",
          np.round((phase['TEAM'] - phase['RECAL']) / nph, 5), flush=True)
