"""
Stage A of the player layer: plate-appearance outcome probabilities from
the batter, the pitcher and the defense, using only games before the date.

Classes are the engine's (`wp_engine.outcome_class`): K, BB, 1B, 2B, 3B,
HR, OUT. "OTHER" rows (baserunning outs that end a half before the PA is
complete, catcher's interference, ...) are not plate-appearance outcomes
and are not counted.

Everything here reads only the seasons it is given (2021-2025 while
sizing; 2026 is never passed in during sizing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wp_engine import outcome_class  # noqa: E402

PA_CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "OUT"]
BIP = ["1B", "2B", "3B", "OUT"]          # ball in play (HR is not: no fielder)
NONBIP = ["K", "BB", "HR"]


def load_pa(states: Path, seasons) -> pd.DataFrame:
    pa = pd.concat([pd.read_parquet(states / f"pa_{s}.parquet") for s in seasons], ignore_index=True)
    pa["date"] = pd.to_datetime(pa["date"])
    pa["cls"] = pa.event_type.map(outcome_class)
    bh = pa.half == "bottom"
    pa["bat_team"] = np.where(bh, pa.home_team, pa.away_team)
    pa["fld_team"] = np.where(bh, pa.away_team, pa.home_team)
    return pa.sort_values(["date", "game_pk", "at_bat_index"]).reset_index(drop=True)


def lineup_slots(pa: pd.DataFrame) -> pd.DataFrame:
    """Batting-order slot of every row, and the lineup as known at that
    moment: for slot offsets 0..8 from the current batter, the batter who
    last occupied that slot up to this row (or, for slots that have not
    batted yet in the game, the first occupant = the lineup card).

    Slot = number of completed plate appearances of the batting team in
    the game before this row, mod 9 (a half that ends on a baserunning out
    leaves the batter's PA incomplete; he leads off next time)."""
    pa = pa.copy()
    done = pa.cls.ne("OTHER") & pa.pa_complete
    key = [pa.game_pk, pa.bat_team]
    pa["pa_idx"] = done.astype(int).groupby(key).cumsum() - done.astype(int)
    pa["slot"] = pa.pa_idx % 9
    nxt = np.zeros((len(pa), 9), dtype=np.int64)
    for (_, _), g in pa.groupby(["game_pk", "bat_team"], sort=False):
        idx = g.index.to_numpy()
        slots = g.slot.to_numpy()
        bats = g.batter.to_numpy()
        first = {}
        for s, b in zip(slots, bats):
            first.setdefault(s, b)
        cur = dict(first)
        for i, s, b in zip(idx, slots, bats):
            cur[s] = b
            nxt[i] = [cur.get((s + k) % 9, -1) for k in range(9)]
    for k in range(9):
        pa[f"next{k}"] = nxt[:, k]
    return pa


def prior_class_counts(pa: pd.DataFrame, who: str, classes: list[str]) -> pd.DataFrame:
    """For every (who, date) in pa: counts of each class on dates strictly
    before that date (over all seasons passed in)."""
    rows = pa[pa.cls.isin(PA_CLASSES) & pa.pa_complete]
    d = pd.crosstab([rows[who], rows.date], rows.cls).reindex(columns=classes, fill_value=0).sort_index()
    thru = d.groupby(level=0).cumsum().reset_index()          # counts through each listed date
    thru["date"] = thru["date"] + pd.Timedelta(days=1)         # usable from the next day on
    q = pa[[who, "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(q, thru.sort_values("date"), on="date", by=who, direction="backward")
    m[classes] = m[classes].fillna(0).astype(float)
    return m


def shrunk(counts: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    n = counts.sum(axis=1, keepdims=True)
    return (counts + k * prior) / (n + k)
