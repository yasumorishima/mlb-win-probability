"""Descriptive (2022-2025, 2026 not read): share of late plate appearances by batters other than the lineup-card batter of that slot, and their prior-record value minus the card batter's (rough linear weights). Run from research/player_layer."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "../..")
import pa_rates as R
S = Path.home() / "claude-scratch/mlbwp/data/states"
pa = R.lineup_slots(R.load_pa(S, [2021, 2022, 2023, 2024, 2025]))
C = R.PA_CLASSES
bat = R.prior_class_counts(pa, "batter", C).set_index(["batter", "date"])
print(C)
# simple value per class (rough linear weights) to rate a batter's prior record
lw = {"K": -0.27, "BB": 0.31, "HBP": 0.33, "1B": 0.47, "2B": 0.77, "3B": 1.04, "HR": 1.40, "OUT": -0.27}
w = np.array([lw.get(c, 0.0) for c in C])
cnt = bat[C].to_numpy(float); n = cnt.sum(1)
lg = cnt.sum(0) / cnt.sum()
k = 200
val = ((cnt + k * lg) / (n + k)[:, None]) @ w
vmap = dict(zip(bat.index, val))
d = pa[pa.season.isin([2022, 2023, 2024, 2025]) & pa.pa_complete & pa.cls.isin(C)].copy()
# lineup card: first occupant of each slot
card = d.groupby(["game_pk", "bat_team", "slot"]).batter.first()
d["card_batter"] = [card[(g, t, s)] for g, t, s in zip(d.game_pk, d.bat_team, d.slot)]
d["sub"] = d.batter != d.card_batter
d["v_act"] = [vmap.get((b, dt), np.nan) for b, dt in zip(d.batter, d.date)]
d["v_card"] = [vmap.get((b, dt), np.nan) for b, dt in zip(d.card_batter, d.date)]
for lo, hi in ((1, 3), (4, 6), (7, 7), (8, 8), (9, 9), (10, 20)):
    x = d[(d.inning >= lo) & (d.inning <= hi)]
    s = x[x["sub"]]
    print(f"inn {lo}-{hi}: PAs {len(x)}, sub share {x['sub'].mean():.3f}, value act-card among subs {np.nanmean(s.v_act - s.v_card):+.4f} (n {len(s)}), per PA overall {np.nanmean(x.v_act - x.v_card):+.5f}")
# by event: pinch hitter vs others? Count first PA of a sub
print("--- innings 7+ by batting team's lead at the PA")
x = d[d.inning >= 7].copy()
x["lead"] = np.where(x.half == "bottom", x.home_score - x.away_score, x.away_score - x.home_score)
x["lb"] = pd.cut(x.lead, [-99, -4, -1, 0, 3, 99], labels=["-4-", "-3..-1", "0", "+1..+3", "+4+"])
for lb, y in x.groupby("lb", observed=True):
    s = y[y["sub"]]
    print(f"lead {lb}: PAs {len(y)}, sub share {y['sub'].mean():.3f}, act-card among subs {np.nanmean(s.v_act - s.v_card):+.4f}, per PA {np.nanmean(y.v_act - y.v_card):+.5f}")
