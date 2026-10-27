"""
Record counts for the v2 sizing (new file: pa_rates.py is frozen with the
2026 test). Earlier seasons can be down-weighted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa_rates import PA_CLASSES


def prior_class_counts_decay(pa: pd.DataFrame, who: str, classes: list[str], lam: float) -> pd.DataFrame:
    """As prior_class_counts, but earlier seasons are weighted lam ** (seasons
    ago): counts of this season on earlier dates (weight 1) plus
    sum over earlier seasons s of lam ** (S - s) * (full season s counts)."""
    rows = pa[pa.cls.isin(PA_CLASSES) & pa.pa_complete]
    season_tot = pd.crosstab([rows[who], rows.season], rows.cls).reindex(columns=classes, fill_value=0)
    # within-season counts before each date
    d = pd.crosstab([rows[who], rows.season, rows.date], rows.cls).reindex(columns=classes, fill_value=0).sort_index()
    thru = d.groupby(level=[0, 1]).cumsum().reset_index()
    thru["date"] = thru["date"] + pd.Timedelta(days=1)
    q = pa[[who, "season", "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(q, thru.sort_values("date"), on="date", by=[who, "season"], direction="backward")
    m[classes] = m[classes].fillna(0).astype(float)
    # earlier seasons, decayed
    st = season_tot.reset_index()
    seasons = sorted(rows.season.unique())
    add = np.zeros((len(m), len(classes)))
    key = st.set_index([who, "season"])[classes]
    for k in range(1, len(seasons)):
        prev = key.reindex(pd.MultiIndex.from_arrays([m[who], m.season - k])).fillna(0).to_numpy()
        add += lam ** k * prev
    m[classes] = m[classes].to_numpy() + add
    return m.drop(columns="season")
