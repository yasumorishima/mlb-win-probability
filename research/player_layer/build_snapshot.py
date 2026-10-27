"""
Snapshot of everything the bench options need at run time, so the app does
not need the plate-appearance tables:

  python research/player_layer/build_snapshot.py --states S --hf HF --ctx CTX --out research/player_layer/published/bench_snapshot.json

Contents (counts of completed plate appearances, classes K BB 1B 2B 3B HR OUT):
  bat / pit           every batter's / pitcher's counts over 2021-2026
  bat_vs / pit_vs     the same split by pitcher hand / batter side
  pen / dfn           each team's 2026 relievers pooled, and its 2026
                      balls-in-play outcomes (1B 2B 3B OUT)
  L, platoon          league class rates and platoon factors (2021-2025,
                      the seasons the engine was fitted on)
  class_tables        P(class | half, state) and per-class transitions
                      (layer.class_tables on the published engine, 2021-2025)
  park                class factors by home team (Savant 3-year factors
                      ending 2026; Athletics / Rays as average)
  hands               bat side and pitch hand
The records end with the 2026 regular season; during 2027 they are not
updated (the screen says so).
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
sys.path.insert(0, str(HERE.parents[1]))
import layer as Ly  # noqa: E402
import pa_rates as R  # noqa: E402
from size_context import ABBR, PF_COLS  # noqa: E402
from size_game import game_context  # noqa: E402
from size_hand import add_hands  # noqa: E402
from wp_engine import Engine  # noqa: E402

C = R.PA_CLASSES


def counts(df, keys):
    t = pd.crosstab([df[k] for k in keys] if len(keys) > 1 else df[keys[0]], df.cls).reindex(columns=C, fill_value=0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pa = R.lineup_slots(R.load_pa(a.states, range(2021, 2027)))
    pa = add_hands(pa, a.ctx)
    pa = game_context(pa)
    done = pa[pa.cls.isin(C) & pa.pa_complete]
    train = pa[pa.season <= 2025]
    eng = Engine.load(HERE.parents[1] / "engine_tables.json")
    ct = Ly.class_tables(eng, train)
    t = train[train.cls.isin(C) & train.pa_complete]
    L = t.cls.value_counts(normalize=True).reindex(C).to_numpy()
    F = pd.crosstab(t.combo, t.cls, normalize="index").reindex(columns=C) / L
    s26 = done[done.season == 2026]
    pen = counts(s26[s26.is_relief], ["fld_team"])
    bip = s26[s26.cls.isin(R.BIP)]
    dfn = pd.crosstab(bip.fld_team, bip.cls).reindex(columns=R.BIP, fill_value=0)
    pf = pd.read_parquet(a.hf / "park_factors.parquet")
    pf["team"] = pf.team.replace({"WAS": "WSH"})
    park = {}
    for r in pf[pf.season == 2026].itertuples():
        if r.team in ("OAK", "TB"):
            continue
        v = []
        for c in C:
            x = getattr(r, PF_COLS[c]) if c in PF_COLS else np.nan
            v.append(1.0 if pd.isna(x) else x / 100.0)
        park[r.team] = v
    hands = pd.read_parquet(a.ctx / "hands.parquet")
    # role: starts and relief appearances in 2025-2026 (relievers only are offered as pitching changes)
    app = pa[pa.season >= 2025].groupby(["game_pk", "fld_team", "pitcher"], sort=False).cur_is_starter.first().reset_index()
    role = app.groupby("pitcher").cur_is_starter.agg(["sum", "size"])
    snap = {
        "classes": C, "bip": R.BIP, "records_through": str(done.date.max().date()),
        "bat": {str(k): v.tolist() for k, v in zip(counts(done, ["batter"]).index, counts(done, ["batter"]).to_numpy())},
        "pit": {str(k): v.tolist() for k, v in zip(counts(done, ["pitcher"]).index, counts(done, ["pitcher"]).to_numpy())},
        "bat_vs": {f"{b}|{h}": v.tolist() for (b, h), v in zip(counts(done, ["batter", "phand"]).index, counts(done, ["batter", "phand"]).to_numpy())},
        "pit_vs": {f"{p}|{h}": v.tolist() for (p, h), v in zip(counts(done, ["pitcher", "bside"]).index, counts(done, ["pitcher", "bside"]).to_numpy())},
        "pen": {str(k): v.tolist() for k, v in zip(pen.index, pen.to_numpy())},
        "dfn": {str(k): v.tolist() for k, v in zip(dfn.index, dfn.to_numpy())},
        "L": L.tolist(), "platoon": {k: v.tolist() for k, v in zip(F.index, F.to_numpy()) if "?" not in k},
        "class_tables": {h: {"P": ct[h]["P"].tolist(), "T": ct[h]["T"].tolist()} for h in ("top", "bottom")},
        "park": park, "team_abbr": {str(k): v for k, v in ABBR.items()},
        "hands": {str(r.player): [r.bat_side, r.pitch_hand] for r in hands.itertuples()},
        "role": {str(k): [int(v["sum"]), int(v["size"])] for k, v in role.iterrows()},   # [starts, games] 2025-2026
    }
    a.out.write_text(json.dumps(snap, separators=(",", ":")))
    print("written", a.out, round(a.out.stat().st_size / 1e6, 2), "MB; records through", snap["records_through"],
          "batters", len(snap["bat"]), "pitchers", len(snap["pit"]), "parks", len(park))


if __name__ == "__main__":
    main()
