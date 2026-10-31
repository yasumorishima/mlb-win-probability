"""Proxy: how much mixture weight the late-inning bullpen model puts on the reliever who
actually started the half (innings 7-9, a reliever not used earlier in the game).
Baseline = PENT rules (60 days, 5 lead states, TAU 1); candidates: recency half-life H
(weights exp(-age/H) on each day's usage), inactive cut X (no relief appearance in the last X days).
2022-2025 only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "../..")
import pa_rates as R
import size_pen_x3 as P
from size_game import game_context
S = Path.home() / "claude-scratch/mlbwp/data/states"
P.WINDOW_DAYS, P.LEAD5, P.TAU = 60, True, 1.0
pa = game_context(R.lineup_slots(R.load_pa(S, [2021, 2022, 2023, 2024, 2025])))
first = pa.groupby(["game_pk", "half", "inning"], sort=False).first().reset_index()
# fielding team's lead at the half's start
first["lead"] = np.where(first.half == "top", first.home_score - first.away_score, first.away_score - first.home_score)
used = {}
for g, gg in pa.groupby("game_pk", sort=False):
    seen = {}
    for (h, i), x in gg.groupby(["half", "inning"], sort=False):
        t = int(x.fld_team.iloc[0])
        used[(g, h, i)] = frozenset(seen.get(t, set()))
        seen.setdefault(t, set()).update(int(p) for p in x.pitcher)
ev = first[first.inning.between(7, 9) & first.is_relief & first.season.isin([2022, 2023, 2024, 2025])].copy()
ev = ev[[int(p) not in used[(g, h, i)] for p, g, h, i in zip(ev.pitcher, ev.game_pk, ev.half, ev.inning)]]
print("events", len(ev))

def daily_usage(s):
    x = pa[(pa.season == s)]
    r = x[x.is_relief & x.inning.isin(range(7, 10))].copy()
    f = x.groupby(["game_pk", "half", "inning"], sort=False)[["home_score", "away_score"]].first()
    r = r.join(f.rename(columns={"home_score": "h0", "away_score": "a0"}), on=["game_pk", "half", "inning"])
    lead = np.where(r.half == "top", r.h0 - r.a0, r.a0 - r.h0)
    r["ls"] = [P.lead_state(int(v)) for v in lead]
    r["ii"] = r.inning - 7
    return r.groupby(["fld_team", "date", "pitcher", "ii", "ls"]).size().rename("n").reset_index()

def last_app(s):
    x = pa[(pa.season == s) & pa.is_relief]
    return x.groupby(["fld_team", "pitcher"]).date.apply(lambda d: np.sort(d.unique())).to_dict()

def score(H=None, X=None, window=60):
    out = []
    for s in (2022, 2023, 2024, 2025):
        du = daily_usage(s)
        tired = P.tired_table(pa[pa.season == s])
        apps = last_app(s)
        by_team = {t: g.sort_values("date") for t, g in du.groupby("fld_team")}
        e = ev[ev.season == s]
        for t, g, h, i, ls_lead, p, dt in zip(e.fld_team, e.game_pk, e.half, e.inning, e.lead, e.pitcher, e.date):
            dd = by_team.get(t)
            if dd is None:
                out.append(np.nan); continue
            w = dd[(dd.date < dt) & (dd.date >= dt - pd.Timedelta(days=window))]
            if X is not None:
                act = set()
                for q in w.pitcher.unique():
                    a = apps.get((t, q))
                    k = np.searchsorted(a, np.datetime64(dt)) if a is not None else 0
                    if k > 0 and a[k - 1] >= np.datetime64(dt - pd.Timedelta(days=X)):
                        act.add(int(q))
            age = (dt - w.date).dt.days.to_numpy()
            wt = w.n.to_numpy(float) * (np.exp(-age / H) if H else 1.0)
            u = {}
            for q, ii, l, v in zip(w.pitcher, w.ii, w.ls, wt):
                u.setdefault(int(q), np.zeros((3, 5)))[ii, l] += v
            avail = {q for q in u if q not in used[(g, h, i)] and q not in tired.get((t, dt), set())}
            if X is not None:
                avail &= act
            W, tot = P.mixture_weights(u, avail, "B")
            ww = W[(i - 7, P.lead_state(int(ls_lead)))]
            sw = sum(ww.values())
            out.append(ww.get(int(p), 0.0) / sw if sw > 0 else np.nan)
    o = np.array(out)
    ok = ~np.isnan(o)
    return {"n": int(ok.sum()), "mean_share_actual": float(o[ok].mean()), "zero_share": float((o[ok] == 0).mean()),
            "mean_log": float(np.log(np.clip(o[ok], 1e-3, 1)).mean())}

import json, sys as _s
arms = {"base60": {}, "H30": {"H": 30}, "H15": {"H": 15}, "X14": {"X": 14}, "X10": {"X": 10}, "H30_w90": {"H": 30, "window": 90}}
for k in _s.argv[1:]:
    print(k, json.dumps(score(**arms[k])), flush=True)
