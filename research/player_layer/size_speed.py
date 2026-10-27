"""
Runner speed: does the runner's previous-season sprint speed predict extra
bases and steal success? 2021-2025, each season scored with a model fitted
on the other four (2026 not read).

  python research/player_layer/size_speed.py --states S --hf HF --out <json>

Events (plate appearances whose class is 1B / 2B, read by base occupancy):
  E1 single, runner on 2nd, 3rd empty           y = a run scored on the play
  E2 single, runner on 1st, 2nd and 3rd empty   y = third base occupied after the play, or a run scored
  E3 double, runner on 1st, 2nd and 3rd empty   y = a run scored
  SB steal attempts of 2nd / 3rd on a pitch     y = stolen base (run table, as research/steal_claude)
Model: logistic on speed_z (previous-season sprint speed, standardised;
missing -> 0 plus a missing flag) and two outs (E1-E3) or target base
(SB); baseline: the same without speed. Metric: log loss, game bootstrap
of the difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pa_rates as R  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
SB_TYPES = {"stolen_base_2b", "stolen_base_3b", "caught_stealing_2b", "caught_stealing_3b"}


def speed_table(hf: Path) -> pd.DataFrame:
    s = pd.read_parquet(hf / "sprint_speed.parquet")[["player_id", "season", "sprint_speed"]]
    mu, sd = s.sprint_speed.mean(), s.sprint_speed.std()
    s = s.assign(season=s.season + 1, speed_z=(s.sprint_speed - mu) / sd)      # usable the following season
    return s[["player_id", "season", "speed_z"]], float(mu), float(sd)


def events(pa: pd.DataFrame) -> pd.DataFrame:
    o = lambda c: pa[c].notna()
    e1 = pa[(pa.cls == "1B") & o("on_2b") & ~o("on_3b")].assign(ev="E1", runner=lambda d: d.on_2b,
                                                                  y=lambda d: (d.runs_on_play >= 1).astype(int))
    e2 = pa[(pa.cls == "1B") & o("on_1b") & ~o("on_2b") & ~o("on_3b")].assign(
        ev="E2", runner=lambda d: d.on_1b, y=lambda d: (d.post_3b.notna() | (d.runs_on_play >= 1)).astype(int))
    e3 = pa[(pa.cls == "2B") & o("on_1b") & ~o("on_2b") & ~o("on_3b")].assign(ev="E3", runner=lambda d: d.on_1b,
                                                                               y=lambda d: (d.runs_on_play >= 1).astype(int))
    out = pd.concat([e1, e2, e3], ignore_index=True)
    out["x2"] = (out.outs == 2).astype(float)
    return out[["ev", "season", "game_pk", "runner", "x2", "y"]]


def steals(states: Path) -> pd.DataFrame:
    run = pd.concat([pd.read_parquet(states / f"run_{s}.parquet") for s in SEASONS], ignore_index=True)
    a = run[(run.row_kind == "outcome") & run.event_type.isin(SB_TYPES) & run.on_pitch].copy()
    a["y"] = a.event_type.str.startswith("stolen_base").astype(int)
    a["x2"] = a.event_type.str.endswith("3b").astype(float)            # target third
    a["ev"] = "SB"
    return a[["ev", "season", "game_pk", "runner", "x2", "y"]]


def fit(X, y):
    def f(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        return np.sum(np.logaddexp(0, z) - y * z), X.T @ (p - y)
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def design(d, with_speed):
    cols = [np.ones(len(d)), d.x2.to_numpy()]
    if with_speed:
        cols += [d.speed_z.to_numpy(), d.miss.to_numpy()]
    return np.column_stack(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sp, mu, sd = speed_table(a.hf)
    pa = R.load_pa(a.states, SEASONS)
    ev = pd.concat([events(pa), steals(a.states)], ignore_index=True)
    ev = ev.merge(sp, left_on=["runner", "season"], right_on=["player_id", "season"], how="left")
    ev["miss"] = ev.speed_z.isna().astype(float)
    ev["speed_z"] = ev.speed_z.fillna(0.0)
    res = {"speed_mean": mu, "speed_sd": sd, "events": {}}
    rng = np.random.default_rng(20261028)
    for name, d in ev.groupby("ev"):
        y = d.y.to_numpy(float)
        ll = {}
        for ws in (False, True):
            p = np.empty(len(d))
            for s in SEASONS:
                tr, te = (d.season != s).to_numpy(), (d.season == s).to_numpy()
                w = fit(design(d[tr], ws), y[tr])
                p[te] = 1 / (1 + np.exp(-(design(d[te], ws) @ w)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll[ws] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        dl = ll[True] - ll[False]
        g = d.game_pk.to_numpy()
        ug, gi = np.unique(g, return_inverse=True)
        wts = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
        bs = (wts @ np.bincount(gi, weights=dl)) / (wts @ np.bincount(gi).astype(float))
        w_all = fit(design(d, True), y)
        res["events"][name] = {"n": int(len(d)), "rate": float(y.mean()), "share_missing_speed": float(d.miss.mean()),
                               "ll_no_speed": float(ll[False].mean()), "ll_speed": float(ll[True].mean()),
                               "diff": float(dl.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               "coef_all_2021_2025": {"intercept": float(w_all[0]), "x2": float(w_all[1]),
                                                      "speed_z": float(w_all[2]), "missing": float(w_all[3])}}
        r = res["events"][name]
        print(name, r["n"], "rate", round(r["rate"], 3), "diff", round(r["diff"], 5), [round(x, 5) for x in r["ci95"]],
              "speed coef", round(r["coef_all_2021_2025"]["speed_z"], 3))
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
