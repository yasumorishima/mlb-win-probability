"""
Independent recomputation from published/pred_2026.parquet (written after the
run; not part of the frozen analysis). A different implementation from
run_2026.py: per-game sums with np.bincount, seed 7, 4,000 game resamples.
Reports the tested differences again and, for the 0.3-0.5 engine band,
intervals that RESULTS.md describes in words.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_parquet(HERE / "published/pred_2026.parquet")
y = d.y.to_numpy()
g = pd.factorize(d.game_pk)[0]
G = g.max() + 1
arms = [c[2:] for c in d.columns if c.startswith("p_")]
P = {a: d["p_" + a].to_numpy() for a in arms}
print("rows", len(d), "games", G)
print("brier", {a: round(float(((P[a] - y) ** 2).mean()), 5) for a in arms})

rng = np.random.default_rng(7)
W = [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(4000)]


def per_game(v, mask=None):
    m = np.ones(len(v), bool) if mask is None else mask
    return np.bincount(g[m], v[m], minlength=G), np.bincount(g[m], minlength=G).astype(float)


def show(name, point, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    print(f"{name}: {point:+.5f} [{lo:+.5f}, {hi:+.5f}] p(>=0) {(1 + (reps >= 0).sum()) / (len(reps) + 1):.4f}")


n_all = np.bincount(g, minlength=G).astype(float)
for a, b in (("TEAM", "RECAL"), ("TEAM", "ENGINE")):
    s, _ = per_game((P[a] - y) ** 2 - (P[b] - y) ** 2)
    show(f"brier {a}-{b}", s.sum() / n_all.sum(), np.array([(w * s).sum() / (w * n_all).sum() for w in W]))

band = (P["ENGINE"] >= 0.3) & (P["ENGINE"] < 0.5)
sy, nb = per_game(y, band)
sums = {a: per_game(P[a], band)[0] for a in ("ENGINE", "RECAL", "TEAM")}


def gap(a, w):
    return (w * sums[a]).sum() / (w * nb).sum() - (w * sy).sum() / (w * nb).sum()


one = np.ones(G)
for a in ("ENGINE", "RECAL", "TEAM"):
    show(f"band gap {a}", gap(a, one), np.array([gap(a, w) for w in W]))
for a, b in (("RECAL", "ENGINE"), ("TEAM", "ENGINE")):
    show(f"band |gap {a}|-|gap {b}|", abs(gap(a, one)) - abs(gap(b, one)),
         np.array([abs(gap(a, w)) - abs(gap(b, w)) for w in W]))
for a in ("RECAL", "TEAM"):
    show(f"band mean {a}-ENGINE", gap(a, one) - gap("ENGINE", one),
         np.array([gap(a, w) - gap("ENGINE", w) for w in W]))
show("band mean TEAM-RECAL", gap("TEAM", one) - gap("RECAL", one),
     np.array([gap("TEAM", w) - gap("RECAL", w) for w in W]))
