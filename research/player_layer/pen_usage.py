"""
Late-inning bullpen usage at run time (bench screen), the same rules as
size_pen.py (variant B):

  usage(team, date)  -> {pitcher: array (3 innings 7/8/9, 4 lead states)}:
                        his relief plate appearances in innings 7-9 for the
                        team over the 30 days before `date` (same season)
  tired(team, date)  -> relievers who pitched on both of the two previous days
  weights(u, avail)  -> {(inning index 0-2, 3 = extra innings, lead state): {pitcher: weight}},
                        usage PAs of the available relievers (pool weight
                        KAPPA / (KAPPA + that))

Data: published/bullpen_usage_<season>.json (build_usage.py), one row per
team, day, pitcher, inning and lead state; built from the states tables.
Lead states (the fielding team's lead when the half begins): 0 behind,
1 tied, 2 ahead by 1-3, 3 ahead by 4 or more.
"""

from __future__ import annotations

import json
from datetime import date as Date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PUB = HERE / "published"
WINDOW_DAYS, TAU, KAPPA = 30, 20.0, 30.0


def lead_state(lead: int) -> int:
    return 0 if lead < 0 else 1 if lead == 0 else 2 if lead <= 3 else 3


@lru_cache(maxsize=8)
def _load(season: int):
    p = PUB / f"bullpen_usage_{season}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    late, relief = {}, {}
    for t, d, pid, ii, ls, n in raw["late"]:
        late.setdefault(int(t), []).append((Date.fromisoformat(d), int(pid), int(ii), int(ls), int(n)))
    for t, d, pid in raw["relief"]:
        relief.setdefault((int(t), Date.fromisoformat(d)), set()).add(int(pid))
    return late, relief


def available(season: int) -> bool:
    return _load(season) is not None


def usage(team: int, day: Date, season: int | None = None) -> dict:
    got = _load(season or day.year)
    if got is None:
        return {}
    lo = day - timedelta(days=WINDOW_DAYS)
    u = {}
    for d, pid, ii, ls, n in got[0].get(int(team), []):
        if lo <= d < day:
            u.setdefault(pid, np.zeros((3, 4)))[ii, ls] += n
    return u


def tired(team: int, day: Date, season: int | None = None) -> set:
    got = _load(season or day.year)
    if got is None:
        return set()
    rel = got[1]
    return rel.get((int(team), day - timedelta(days=1)), set()) & rel.get((int(team), day - timedelta(days=2)), set())


def weights(u: dict, avail, variant: str = "B") -> tuple[dict, float]:
    """Same as size_pen.mixture_weights."""
    W = {}
    avail = [p for p in avail if p in u]
    tot = float(sum(u[p].sum() for p in avail))
    for ii in range(4):
        rows = {p: (u[p] if ii == 3 else u[p][ii:ii + 1]) for p in avail}
        tot_i = sum(x.sum() for x in rows.values())
        for ls in range(4):
            w = {}
            for p, x in rows.items():
                v = x.sum() if variant == "A" else x[:, ls].sum() + TAU * x.sum() / max(tot_i, 1.0)
                if v > 0:
                    w[p] = v
            W[(ii, ls)] = w
    return W, tot
