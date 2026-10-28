"""
Pre-registered test of PEN_RECAL on 2026 (PREREG_PEN.md).

PEN_RECAL = the player layer with the late-inning bullpen by usage (size_pen.py,
variant B), recalibrated with a logistic layer (terms L, L r, L r^2, r, r^2 as
size_stack.py P_RECAL) fitted on the 2022-2025 out-of-fold predictions only.

  fit     : python test_pen_2026.py fit   --states S --pen P --out-coef published/pen_recal_coef.json
            (P = dir with size_pen.py penB_2022..2025.parquet; reads 2022-2025 outcomes only)
  dry     : python test_pen_2026.py dry   --states S --pen P
            (gate: refit on 2022-2024, score 2025; must equal size_stack's P_RECAL 2025 Brier; no 2026 read)
  predict : python test_pen_2026.py predict --states S --hf HF --out W
            (reads published/penB_2026_part{0..3}.parquet; no outcome read)
  score   : python test_pen_2026.py score --states S --hf HF --out W   (once; needs Status: FROZEN)
  score_dry: python test_pen_2026.py score_dry --states S --pen P --game G
            (rehearsal of the scoring path on 2025 sizing rows: PEN_RECAL fitted on 2022-2024 vs the
             out-of-fold player layer v1 and engine of size_game.py; no 2026 read)

2026 predictions: size_pen.py --season 2026 --variant B --load-seasons 2021,2022,2023,2024,2025,2026
--part K/4 (tables fitted on 2021-2025, records / usage from earlier dates only).
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
from size_stack import design, fit  # noqa: E402

SEED, N_BOOT = 20261030, 2000
N_2026 = 184327
TERMS = ["L", "Lr", "Lr2", "r", "r2"]
COEF = HERE / "published/pen_recal_coef.json"
TEAM_FILE = ROOT / "research/wp_layer/published/pred_2026.parquet"
PLAYER_FILES = [HERE / f"published/pred_2026_part{k}.parquet" for k in range(4)]
BLEND_FILE = HERE / "published/blend_2026.parquet"
PEN26_FILES = [HERE / f"published/penB_2026_part{k}.parquet" for k in range(4)]
PARK_MD5 = "3f1c06008ec1fec5012a8f50338b9687"     # HF park_factors.parquet, same file as test_2026.py
INPUT_MD5 = {
    "pa_2021.parquet": "10947f7add3b8821ddaa70705b845058",
    "pa_2022.parquet": "bbacd8f9703d5922d615903ebd693ca3",
    "pa_2023.parquet": "c8e4f89027179b1b09465a3d73580011",
    "pa_2024.parquet": "7d7f79769fae15f0c431d6389b4624ea",
    "pa_2025.parquet": "9d82f22f73ddd0189714b5c277a315ef",
    "pa_2026.parquet": "1c54a8d674846aac475aa87ce27fcc3c",
    "pitch_2021.parquet": "74291c933f214f2e3a52353d4ef76fa3",
    "pitch_2022.parquet": "7a5106e73c96e1a55e3928d8976a4fed",
    "pitch_2023.parquet": "c2c200babacced32229c537ff31ead4d",
    "pitch_2024.parquet": "1e384f82963fa2790957a24b14f37ee8",
    "pitch_2025.parquet": "e5ee8adff8b0b4126a0deddb97cab5ee",
    "pitch_2026.parquet": "d21f7b296296216f0a5b51357869b8fc",
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def check_inputs(states: Path, hf: Path | None = None) -> None:
    bad = {k: md5(states / k) for k, v in INPUT_MD5.items() if md5(states / k) != v}
    if hf is not None and md5(hf / "park_factors.parquet") != PARK_MD5:
        bad["park_factors.parquet"] = md5(hf / "park_factors.parquet")
    if bad:
        sys.exit(f"input md5 mismatch: {bad}")


def frozen(required: bool) -> dict:
    text = (HERE / "PREREG_PEN.md").read_text()
    listed = re.findall(r"^([0-9a-f]{32})  (\S+)$", text, flags=re.M)
    wrong = {p: md5(ROOT / p) for m, p in listed if md5(ROOT / p) != m}
    is_frozen = re.search(r"^Status: FROZEN", text, flags=re.M) is not None
    if required and (not is_frozen or not listed or wrong):
        sys.exit(f"frozen check failed: FROZEN={is_frozen} listed={len(listed)} mismatched={wrong}")
    return {"status_frozen": is_frozen, "n_listed": len(listed), "mismatched": wrong,
            "prereg_md5": md5(HERE / "PREREG_PEN.md"), "script_md5": md5(Path(__file__))}


def with_r(d: pd.DataFrame) -> pd.DataFrame:
    """r = share of regulation outs still to be played (floored), as size_stack.py."""
    total = 6 * d.scheduled_innings
    done = ((d.inning - 1) * 2 + (d.half == "bottom")) * 3 + d.outs
    d = d.copy()
    d["r"] = np.maximum(1 - done / total, 3 / total)
    for c in ("d_prev", "d_cur", "diff"):      # size_stack.design builds every column; unused by TERMS
        d[c] = 0.0
    return d


def sizing_rows(states: Path, pen: Path, seasons) -> pd.DataFrame:
    rows = pd.concat([pd.read_parquet(pen / f"penB_{s}.parquet") for s in seasons], ignore_index=True)
    cols = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings", "home_won"]
    pa = pd.concat([pd.read_parquet(states / f"pa_{s}.parquet", columns=cols) for s in seasons], ignore_index=True)
    d = rows.merge(pa, on=["game_pk", "at_bat_index"], how="left")
    if d[cols].isna().any().any() or len(d) != len(pa):
        sys.exit("sizing rows do not match the plate appearances")
    return with_r(d.rename(columns={"wp_new": "wp_game"}))


def apply(coef: dict, d: pd.DataFrame) -> np.ndarray:
    return 1 / (1 + np.exp(-(design(d, "wp_game", coef["terms"]) @ np.asarray(coef["coef"]))))


def cmd_fit(a) -> None:
    check_inputs(a.states)
    d = sizing_rows(a.states, a.pen, [2022, 2023, 2024, 2025])
    w = fit(design(d, "wp_game", TERMS), d.home_won.to_numpy(float))
    out = {"base": "wp_game (size_pen.py variant B)", "terms": TERMS, "coef": [float(x) for x in w],
           "fitted_on": [2022, 2023, 2024, 2025], "n": int(len(d)),
           "pen_md5": {s: md5(a.pen / f"penB_{s}.parquet") for s in (2022, 2023, 2024, 2025)}}
    a.out_coef.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_dry(a) -> None:
    check_inputs(a.states)
    tr = sizing_rows(a.states, a.pen, [2022, 2023, 2024])
    te = sizing_rows(a.states, a.pen, [2025])
    w = fit(design(tr, "wp_game", TERMS), tr.home_won.to_numpy(float))
    p = apply({"terms": TERMS, "coef": w}, te)
    y = te.home_won.to_numpy(float)
    print("dry 2025: PEN", float(((te.wp_game.to_numpy() - y) ** 2).mean()), "PEN_RECAL", float(((p - y) ** 2).mean()))


def pen_2026_rows() -> pd.DataFrame:
    coef = json.loads(COEF.read_text())
    pen = pd.concat([pd.read_parquet(f) for f in PEN26_FILES], ignore_index=True)
    base = pd.concat([pd.read_parquet(f) for f in PLAYER_FILES], ignore_index=True)
    d = base.merge(pen[["game_pk", "at_bat_index", "wp_new"]], on=["game_pk", "at_bat_index"], how="left")
    if len(pen) != N_2026 or len(d) != N_2026 or d.wp_new.isna().any() or pen.duplicated(["game_pk", "at_bat_index"]).any():
        sys.exit("2026 pen predictions incomplete")
    d = with_r(d.rename(columns={"wp_new": "wp_game"}))
    res = d[["game_pk", "at_bat_index", "wp_player", "wp_engine"]].copy()
    res["wp_pen"] = d.wp_game
    res["p_PEN_RECAL"] = apply(coef, d)
    if not np.isfinite(res.drop(columns=["game_pk", "at_bat_index"]).to_numpy()).all():
        sys.exit("non-finite predictions")
    return res


def cmd_predict(a) -> None:
    check_inputs(a.states, a.hf)
    res = pen_2026_rows()
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out / "pen_2026.parquet", index=False)
    print("predictions", len(res), {f.name: md5(f) for f in PEN26_FILES})


def cmd_score(a) -> None:
    check_inputs(a.states, a.hf)
    res = {"frozen": frozen(required=True),
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    d = pd.read_parquet(a.out / "pen_2026.parquet")
    res["pred_md5"] = md5(a.out / "pen_2026.parquet")
    res["pen_part_md5"] = {f.name: md5(f) for f in PEN26_FILES}
    again = pen_2026_rows()          # the predictions must come from the frozen parts and coefficients
    if not (again[["game_pk", "at_bat_index"]].equals(d[["game_pk", "at_bat_index"]])
            and np.array_equal(again.p_PEN_RECAL.to_numpy(), d.p_PEN_RECAL.to_numpy())
            and np.array_equal(again.wp_pen.to_numpy(), d.wp_pen.to_numpy())):
        sys.exit("pen_2026.parquet does not equal the frozen parts + coefficients")
    d = d.merge(pd.read_parquet(TEAM_FILE, columns=["game_pk", "at_bat_index", "p_TEAM"]), on=["game_pk", "at_bat_index"], how="left")
    d = d.merge(pd.read_parquet(BLEND_FILE, columns=["game_pk", "at_bat_index", "p_BLEND"]), on=["game_pk", "at_bat_index"], how="left")
    y_tab = pd.read_parquet(a.states / "pa_2026.parquet", columns=["game_pk", "at_bat_index", "home_won"])
    d = d.merge(y_tab, on=["game_pk", "at_bat_index"], how="left")
    if len(d) != N_2026 or d[["p_TEAM", "p_BLEND", "home_won"]].isna().any().any():
        sys.exit("gate failed")
    if not np.isfinite(d[["p_PEN_RECAL", "wp_pen", "p_TEAM", "wp_player", "wp_engine", "p_BLEND"]].to_numpy()).all():
        sys.exit("non-finite predictions")
    y = d.home_won.to_numpy(float)
    arms = {"PEN_RECAL": d.p_PEN_RECAL, "TEAM": d.p_TEAM, "PLAYER": d.wp_player, "ENGINE": d.wp_engine,
            "PEN": d.wp_pen, "BLEND": d.p_BLEND}
    se = {k: (v.to_numpy() - y) ** 2 for k, v in arms.items()}
    res["arms"] = {}
    for k, v in arms.items():
        v = v.to_numpy(); p = np.clip(v, 1e-4, 1 - 1e-4)
        band = (v >= 0.3) & (v < 0.5)
        res["arms"][k] = {"brier": float(se[k].mean()), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                          "band_0.3_0.5_gap_pt": float(100 * (v[band].mean() - y[band].mean()))}
    tests, holm, ng = run_tests(d.game_pk.to_numpy(), se, (("H1_PEN_RECAL_minus_TEAM", "TEAM"), ("H2_PEN_RECAL_minus_PLAYER", "PLAYER")))
    res.update({"n_rows": int(len(d)), "n_games": ng, "tests": tests, "holm_adjusted_p": holm,
                "supported": {h: bool(v <= 0.05) for h, v in holm.items()}})
    (a.out / "results_pen_2026.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def run_tests(g, se, pairs):
    ug, gi = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)
    cnt = w @ np.bincount(gi).astype(float)
    tests = {}
    for h, other in pairs:
        dl = se["PEN_RECAL"] - se[other]
        bs = (w @ np.bincount(gi, weights=dl)) / cnt
        tests[h] = {"mean": float(dl.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "p_one_sided": float((1 + (bs >= 0).sum()) / (N_BOOT + 1))}
    pv = {k: v["p_one_sided"] for k, v in tests.items()}
    order = sorted(pv, key=pv.get)
    holm, prev = {}, 0.0
    for i, h in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - i) * pv[h]))
        holm[h] = prev
    return tests, holm, int(len(ug))


def cmd_score_dry(a) -> None:
    """Rehearsal on 2025 (sizing rows only): PEN_RECAL fitted on 2022-2024 vs the
    out-of-fold player layer v1 (TEAM stand-in) and engine (size_game.py)."""
    check_inputs(a.states)
    tr = sizing_rows(a.states, a.pen, [2022, 2023, 2024])
    te = sizing_rows(a.states, a.pen, [2025])
    w = fit(design(tr, "wp_game", TERMS), tr.home_won.to_numpy(float))
    te["p_PEN_RECAL"] = apply({"terms": TERMS, "coef": w}, te)
    v1 = pd.read_parquet(a.game / "game_2025.parquet", columns=["game_pk", "at_bat_index", "wp_game", "wp_league"])
    te = te.merge(v1.rename(columns={"wp_game": "wp_v1"}), on=["game_pk", "at_bat_index"], how="left")
    y = te.home_won.to_numpy(float)
    se = {"PEN_RECAL": (te.p_PEN_RECAL.to_numpy() - y) ** 2, "PLAYER_v1": (te.wp_v1.to_numpy() - y) ** 2,
          "ENGINE_oof": (te.wp_league.to_numpy() - y) ** 2}
    tests, holm, ng = run_tests(te.game_pk.to_numpy(), se, (("PEN_RECAL_minus_ENGINE_oof", "ENGINE_oof"),
                                                            ("PEN_RECAL_minus_PLAYER_v1", "PLAYER_v1")))
    print(json.dumps({"n_rows": len(te), "n_games": ng, "brier": {k: float(v.mean()) for k, v in se.items()},
                      "tests": tests, "holm": holm}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "dry", "predict", "score", "score_dry"])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--pen", type=Path, help="size_pen.py sizing output (penB_2022..2025)")
    ap.add_argument("--hf", type=Path, help="dir with HF park_factors.parquet (md5-checked)")
    ap.add_argument("--game", type=Path, help="size_game.py output (game_2025.parquet) for score_dry")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-coef", type=Path, default=COEF)
    a = ap.parse_args()
    {"fit": cmd_fit, "dry": cmd_dry, "predict": cmd_predict, "score": cmd_score, "score_dry": cmd_score_dry}[a.mode](a)


if __name__ == "__main__":
    main()
