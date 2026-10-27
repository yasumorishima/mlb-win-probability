"""
Pre-registered test of BLEND on 2026 (PREREG_BLEND.md).

  python research/player_layer/test_blend_2026.py predict --states S --out W     (no outcome read)
  python research/player_layer/test_blend_2026.py score   --states S --out W     (once)

Inputs (md5-checked): blend_coef.json (fitted on 2022-2025 only), the
frozen player-layer predictions for 2026 (published/pred_2026_part*.parquet,
columns wp_player / wp_engine), the frozen team layer's p_TEAM
(research/wp_layer/published/pred_2026.parquet, prediction column only
until the score step), team strength from layer_features.team_strength
(games on earlier dates).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "wp_layer"))
from layer_features import game_table, load_pa, team_strength  # noqa: E402
from size_stack import design, logit  # noqa: E402

SEED, N_BOOT = 20261029, 2000
TEAM_FILE = ROOT / "research/wp_layer/published/pred_2026.parquet"
INPUT_MD5 = {
    "pa_2025.parquet": "9d82f22f73ddd0189714b5c277a315ef",
    "pa_2026.parquet": "1c54a8d674846aac475aa87ce27fcc3c",
    "pred_2026.parquet": "c9e35870ab157190d2d81b1f26b23b87",
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def verify(states: Path, required: bool) -> dict:
    bad = {k: md5(p) for k, p in {"pa_2025.parquet": states / "pa_2025.parquet", "pa_2026.parquet": states / "pa_2026.parquet",
                                  "pred_2026.parquet": TEAM_FILE}.items() if md5(p) != INPUT_MD5[k]}
    if bad:
        sys.exit(f"input md5 mismatch: {bad}")
    text = (HERE / "PREREG_BLEND.md").read_text()
    listed = re.findall(r"^([0-9a-f]{32})  (\S+)$", text, flags=re.M)
    wrong = {p: md5(ROOT / p) for m, p in listed if md5(ROOT / p) != m}
    frozen = re.search(r"^Status: FROZEN", text, flags=re.M) is not None
    if required and (not frozen or not listed or wrong):
        sys.exit(f"frozen check failed: FROZEN={frozen} listed={len(listed)} mismatched={wrong}")
    return {"status_frozen": frozen, "n_listed": len(listed), "mismatched": wrong, "prereg_md5": md5(HERE / "PREREG_BLEND.md"),
            "script_md5": md5(Path(__file__))}


def predict(a) -> None:
    verify(a.states, required=False)
    coef = json.loads((HERE / "published/blend_coef.json").read_text())
    pred = pd.concat([pd.read_parquet(HERE / f"published/pred_2026_part{k}.parquet") for k in range(4)], ignore_index=True)
    pa = load_pa(a.states, [2025, 2026])
    ts = team_strength(game_table(pa))
    d = pred.join(ts, on="game_pk")
    total = 6 * d.scheduled_innings
    done = ((d.inning - 1) * 2 + (d.half == "bottom")) * 3 + d.outs
    d["r"] = np.maximum(1 - done / total, 3 / total)
    d["wp_game"], d["wp_league"] = d.wp_player, d.wp_engine
    if d[["d_prev", "d_cur", "r", "wp_game", "wp_league"]].isna().any().any() or len(d) != 184327:
        sys.exit("missing inputs")
    out = {}
    for name in ("P_RECAL", "E_TEAM"):
        c = coef[name]
        out[name] = 1 / (1 + np.exp(-(design(d, c["base"], c["terms"]) @ np.asarray(c["coef"]))))
    w = coef["weight_player"]
    out["BLEND"] = 1 / (1 + np.exp(-(w * logit(out["P_RECAL"]) + (1 - w) * logit(out["E_TEAM"]))))
    res = d[["game_pk", "at_bat_index", "wp_player", "wp_engine"]].copy()
    for k, v in out.items():
        res[f"p_{k}"] = v
    if not np.isfinite(res.drop(columns=["game_pk", "at_bat_index"]).to_numpy()).all():
        sys.exit("non-finite predictions")
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out / "blend_2026.parquet", index=False)
    print("predictions", len(res))


def score(a) -> None:
    res = {"frozen": verify(a.states, required=True),
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    d = pd.read_parquet(a.out / "blend_2026.parquet")
    team = pd.read_parquet(TEAM_FILE, columns=["game_pk", "at_bat_index", "p_TEAM"])
    d = d.merge(team, on=["game_pk", "at_bat_index"], how="left")
    if len(d) != 184327 or d.p_TEAM.isna().any():
        sys.exit("gate failed")
    y_tab = load_pa(a.states, [2026])[["game_pk", "at_bat_index", "home_won"]]
    d = d.merge(y_tab, on=["game_pk", "at_bat_index"], how="left")
    y = d.home_won.to_numpy(float)
    arms = {"BLEND": d.p_BLEND, "TEAM": d.p_TEAM, "PLAYER": d.wp_player, "ENGINE": d.wp_engine,
            "P_RECAL": d.p_P_RECAL, "E_TEAM": d.p_E_TEAM}
    se = {k: (v.to_numpy() - y) ** 2 for k, v in arms.items()}
    res["arms"] = {}
    for k, v in arms.items():
        v = v.to_numpy(); p = np.clip(v, 1e-4, 1 - 1e-4)
        band = (v >= 0.3) & (v < 0.5)
        res["arms"][k] = {"brier": float(se[k].mean()), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                          "band_0.3_0.5_gap_pt": float(100 * (v[band].mean() - y[band].mean()))}
    g = d.game_pk.to_numpy()
    ug, gi = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)
    cnt = w @ np.bincount(gi).astype(float)
    tests = {}
    for h, other in (("H1_BLEND_minus_TEAM", "TEAM"), ("H2_BLEND_minus_PLAYER", "PLAYER")):
        dl = se["BLEND"] - se[other]
        bs = (w @ np.bincount(gi, weights=dl)) / cnt
        tests[h] = {"mean": float(dl.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "p_one_sided": float((1 + (bs >= 0).sum()) / (N_BOOT + 1))}
    pv = {k: v["p_one_sided"] for k, v in tests.items()}
    order = sorted(pv, key=pv.get)
    holm, prev = {}, 0.0
    for i, h in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - i) * pv[h]))
        holm[h] = prev
    res.update({"tests": tests, "holm_adjusted_p": holm, "supported": {h: bool(v <= 0.05) for h, v in holm.items()}})
    (a.out / "results_blend_2026.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "score"])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    predict(a) if a.mode == "predict" else score(a)


if __name__ == "__main__":
    main()
