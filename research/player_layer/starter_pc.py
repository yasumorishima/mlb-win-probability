"""
Starter-length table by pitch count (sizing candidate for size_pen_x.py --pc-edges).

size_game2.starter_table gives P(the starter begins inning i | he is pitching now in
inning c with batters-faced bucket b). Here the bucket is taken from any per-row
count column (pitches thrown so far instead of batters faced). With
col="bf_before" and BF_EDGES it equals size_game2.starter_table (gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bucket(x, edges) -> int:
    return int(np.searchsorted(edges, x, side="right") - 1)


def starter_table_by(tr: pd.DataFrame, col: str, edges) -> np.ndarray:
    nb = len(edges) - 1
    first = tr.groupby(["game_pk", "fld_team", "inning"], sort=False).pitcher.first().reset_index()
    first = first.merge(tr.groupby(["game_pk", "fld_team"]).pitcher.first().rename("starter").reset_index(),
                        on=["game_pk", "fld_team"])
    began = {(g, t, i) for g, t, i, p, s in first.itertuples(index=False) if p == s}
    num = np.zeros((13, nb, 13)); den = np.zeros((13, nb))
    rows = tr[tr.cur_is_starter & (tr.inning <= 12)]
    for g, t, c, x in zip(rows.game_pk, rows.fld_team, rows.inning, rows[col]):
        b = bucket(x, edges)
        den[c, b] += 1
        for i in range(c + 1, 13):
            if (g, t, i) in began:
                num[c, b, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        q = num / den[:, :, None]
    return np.nan_to_num(q)


def pitch_counts(pa: pd.DataFrame, pitch: pd.DataFrame) -> pd.DataFrame:
    """Adds n_pitch (pitches in the plate appearance), pc_before (pitches the current
    pitcher threw in this game before it) and bat_pitcher_pc (the same for the batting
    team's pitcher, before this row). All pitches of a plate appearance are credited to
    the pitcher on record for it (a mid-PA change is rare). pa must be in game order."""
    npit = pitch.groupby(["game_pk", "at_bat_index"]).size()
    idx = pd.MultiIndex.from_arrays([pa.game_pk.to_numpy(), pa.at_bat_index.to_numpy()])
    n = npit.reindex(idx).fillna(0).to_numpy(np.int64)
    pa = pa.copy()
    pa["n_pitch"] = n
    pa["pc_before"] = pa.groupby(["game_pk", "pitcher"]).n_pitch.cumsum().to_numpy() - n
    out = np.zeros(len(pa), dtype=np.int64)
    g_prev, cnt = None, {}
    for i, (g, p, bp, k) in enumerate(zip(pa.game_pk.to_numpy(), pa.pitcher.to_numpy(),
                                          pa.bat_pitcher.to_numpy(), n)):
        if g != g_prev:
            g_prev, cnt = g, {}
        out[i] = cnt.get(bp, 0)
        cnt[p] = cnt.get(p, 0) + k
    pa["bat_pitcher_pc"] = out
    return pa
