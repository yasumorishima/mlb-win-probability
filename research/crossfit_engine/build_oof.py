"""
Leave-one-season-out engines for 2021-2025 (CLAUDE.md, next task 1).

For each season s in 2021-2025 the engine tables are fitted on the other
four seasons with the unchanged `wp_engine.fit_tables`, and home win
probability is predicted before every plate appearance of s. The result is
an out-of-fold WP for every training-season PA: the baseline any new layer
(player layer, recalibration) is fitted against, so that the layer learns
the engine's out-of-sample error instead of its in-sample fit.

Gate first: the all-seasons refit must reproduce `engine_tables.json`
(every table equal; `league_swing`, a sampled mean, within 1e-12).

  python research/crossfit_engine/build_oof.py --states <dir> --out <dir>

Writes <out>/oof_wp.parquet (game_pk, at_bat_index, season, wp_oof,
wp_full = the published engine's WP) and <out>/summary.json. 2026 is not
read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wp_engine import Engine, _occ, fit_tables  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]


def load(states: Path, kind: str, seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(states / f"{kind}_{s}.parquet") for s in seasons], ignore_index=True)


def predict(eng: Engine, pa: pd.DataFrame) -> np.ndarray:
    return np.array([eng.wp(r.inning, r.half, r.outs, (_occ(r.on_1b), _occ(r.on_2b), _occ(r.on_3b)),
                            r.home_score - r.away_score, r.scheduled_innings)
                     for r in pa.itertuples(index=False)])


def gate(pa: pd.DataFrame, pitch: pd.DataFrame) -> dict:
    pub = json.loads((ROOT / "engine_tables.json").read_text())
    t = fit_tables(pa, pitch)
    same = {k: t[k] == pub[k] for k in t}
    if not all(same.values()):
        sys.exit(f"gate failed: refit differs from engine_tables.json in {[k for k, v in same.items() if not v]}")
    return {"tables_equal": sorted(same), "n_pa": t["n_pa"]}


def score(p, y) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def band(p, y, lo, hi) -> dict:
    m = (p >= lo) & (p < hi)
    return {"n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean()),
            "bias_pt": float(100 * (p[m].mean() - y[m].mean()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pa = load(a.states, "pa", SEASONS)
    pitch = load(a.states, "pitch", SEASONS)
    summary = {"gate": gate(pa, pitch)}
    print("gate ok:", summary["gate"])

    full = Engine.load(ROOT / "engine_tables.json")
    parts = []
    for s in SEASONS:
        tr = [x for x in SEASONS if x != s]
        eng = Engine(fit_tables(pa[pa.season.isin(tr)], pitch[pitch.game_pk.isin(pa[pa.season.isin(tr)].game_pk)]))
        te = pa[pa.season == s]
        parts.append(pd.DataFrame({"game_pk": te.game_pk.to_numpy(), "at_bat_index": te.at_bat_index.to_numpy(),
                                   "season": s, "home_won": te.home_won.to_numpy(),
                                   "wp_oof": predict(eng, te), "wp_full": predict(full, te)}))
        print("season", s, "done")
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(a.out / "oof_wp.parquet", index=False)

    y = oof.home_won.to_numpy().astype(float)
    for col in ("wp_oof", "wp_full"):
        p = oof[col].to_numpy()
        summary[col] = {"all": score(p, y), "band_0.3_0.5": band(p, y, 0.3, 0.5),
                        "by_season": {int(s): score(oof[oof.season == s][col].to_numpy(),
                                                    oof[oof.season == s].home_won.to_numpy().astype(float))
                                      for s in SEASONS},
                        "band_by_season": {int(s): band(oof[oof.season == s][col].to_numpy(),
                                                        oof[oof.season == s].home_won.to_numpy().astype(float), 0.3, 0.5)
                                           for s in SEASONS}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k]["all"] | {"band": summary[k]["band_0.3_0.5"]} for k in ("wp_oof", "wp_full")}, indent=1))


if __name__ == "__main__":
    main()
