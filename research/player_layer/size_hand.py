"""
Stage A, handedness (platoon) on top of V5 + park: 2022-2025, league
tables from the other four of 2021-2025, records causal. 2026 not read.

  python research/player_layer/size_hand.py --states <dir> --hf <dir> --ctx <dir with hands.parquet> --out <json>

Effective batter side: switch hitters bat opposite the pitcher.
  league : x f(combo)^beta, f = P(class | combo) / P(class) on the
           training seasons, combo = (batter side, pitcher hand)
  own    : league factor, plus the batter's own split: his rates against
           this pitcher hand vs his overall rates, shrunk (k = kh) toward
           the league split; the same for the pitcher against this
           batter side
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
import size_pa as S  # noqa: E402
from size_context import ABBR, PF_COLS  # noqa: E402

C = R.PA_CLASSES


def add_hands(pa: pd.DataFrame, ctx: Path) -> pd.DataFrame:
    h = pd.read_parquet(ctx / "hands.parquet").set_index("player")
    pa = pa.copy()
    pa["phand"] = pa.pitcher.map(h.pitch_hand)
    side = pa.batter.map(h.bat_side)
    pa["bside"] = np.where(side == "S", np.where(pa.phand == "R", "L", "R"), side)
    pa["combo"] = pa.bside.fillna("?") + "v" + pa.phand.fillna("?")
    return pa


def split_counts(pa: pd.DataFrame, who: str, by: str) -> pd.DataFrame:
    """Class counts of `who` against each value of `by`, dates strictly before."""
    rows = pa[pa.cls.isin(C) & pa.pa_complete]
    d = pd.crosstab([rows[who], rows[by], rows.date], rows.cls).reindex(columns=C, fill_value=0).sort_index()
    thru = d.groupby(level=[0, 1]).cumsum().reset_index()
    thru["date"] = thru["date"] + pd.Timedelta(days=1)
    q = pa[[who, by, "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(q, thru.sort_values("date"), on="date", by=[who, by], direction="backward")
    m[C] = m[C].fillna(0).astype(float)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pa = R.load_pa(a.states, S.SEASONS)
    pa = add_hands(pa[pa.cls.isin(C) & pa.pa_complete].reset_index(drop=True), a.ctx)
    for who in ("batter", "pitcher"):
        cnt = R.prior_class_counts(pa, who, C)
        pa = pa.merge(cnt.rename(columns={c: f"{who}_{c}" for c in C}), on=[who, "date"], how="left")
    bs = split_counts(pa, "batter", "phand")
    pa = pa.merge(bs.rename(columns={c: f"bsplit_{c}" for c in C}), on=["batter", "phand", "date"], how="left")
    ps = split_counts(pa, "pitcher", "bside")
    pa = pa.merge(ps.rename(columns={c: f"psplit_{c}" for c in C}), on=["pitcher", "bside", "date"], how="left")
    dc = S.defense_counts(pa)
    pa = pa.merge(dc.rename(columns={c: f"def_{c}" for c in R.BIP}), on=["fld_team", "season", "date"], how="left")
    pa["sk"] = S.state_key(pa)

    pf = pd.read_parquet(a.hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    pf = pf.assign(season=pf.season + 1).set_index(["team", "season"])
    j = pf.reindex(pd.MultiIndex.from_arrays([pa.home_team.map(ABBR), pa.season]))
    PF = np.ones((len(pa), len(C)))
    for c, col in PF_COLS.items():
        PF[:, C.index(c)] = j[col].to_numpy() / 100.0
    PF[((pa.home_team.map(ABBR).isin(["OAK", "TB"])) & (pa.season == 2025)).to_numpy()] = np.nan
    PF[np.isnan(PF)] = 1.0

    y = pa.cls.map({c: i for i, c in enumerate(C)}).to_numpy()
    res = {"missing_hand_rows": int(pa.combo.str.contains(r"\?").sum()), "combo_share": pa.combo.value_counts(normalize=True).round(4).to_dict()}
    folds = {}
    for s in S.SCORE:
        tr = pa[pa.season != s]
        st, L = S.league_tables(tr)
        f = pd.crosstab(tr.combo, tr.cls, normalize="index").reindex(columns=C) / L
        m = (pa.season == s).to_numpy()
        d = pa[m]
        M = Ly.multipliers(d[[f"batter_{c}" for c in C]].to_numpy(), d[[f"pitcher_{c}" for c in C]].to_numpy(),
                           d[[f"def_{c}" for c in R.BIP]].to_numpy(), L)
        base = st.reindex(d.sk).to_numpy() * M * PF[m] ** 0.5
        F = f.reindex(d.combo).fillna(1.0).to_numpy()
        # own split: (rates vs this hand) / (overall rates), each shrunk; target = league factor
        def own(prefix_split, prefix_all, k, d=d, F=F, L=L):
            sp = d[[f"{prefix_split}_{c}" for c in C]].to_numpy()
            al = d[[f"{prefix_all}_{c}" for c in C]].to_numpy()
            a_r = R.shrunk(al, L, 200)
            s_r = (sp + k * a_r * F) / (sp.sum(axis=1, keepdims=True) + k)
            return s_r / a_r / F
        folds[s] = dict(base=base, F=F, own=own, y=y[m], g=d.game_pk.to_numpy())

    def ll(beta, kh=None, per_row=False):
        out = []
        for f in folds.values():
            p = f["base"] * f["F"] ** beta
            if kh is not None:
                p = p * f["own"]("bsplit", "batter", kh) * f["own"]("psplit", "pitcher", kh)
            p = p / p.sum(axis=1, keepdims=True)
            out.append(-np.log(p[np.arange(len(f["y"])), f["y"]]))
        v = np.concatenate(out)
        return v if per_row else float(v.mean())

    res["grid"] = [{"beta": b, "ll": ll(b)} for b in (0.0, 0.5, 1.0, 1.5)]
    bb = min(res["grid"], key=lambda x: x["ll"])["beta"]
    res["grid_own"] = [{"beta": bb, "kh": k, "ll": ll(bb, k)} for k in (500, 1000, 2000, 4000, 8000, 16000)]
    bk = min(res["grid_own"], key=lambda x: x["ll"])["kh"]
    g = np.concatenate([f["g"] for f in folds.values()])
    ug, gi = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(20261027)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=2000)
    cnt = np.bincount(gi).astype(float)
    base = ll(0.0, per_row=True)
    res["vs_V5_park"] = {}
    for name, (b, k) in {"league_platoon": (bb, None), "plus_own_split": (bb, bk)}.items():
        dlt = ll(b, k, per_row=True) - base
        bs_ = (w @ np.bincount(gi, weights=dlt)) / (w @ cnt)
        res["vs_V5_park"][name] = {"beta": b, "kh": k, "mean": float(dlt.mean()),
                                   "ci95": [float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))]}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
