"""
Pre-registered test of PENT_RECAL on 2026 (PREREG_PENT.md).

PENT_RECAL = PENR_RECAL (60-day usage, five lead states, KAPPA 0.25) with the lead-state
smoothing TAU 1 instead of 20 (size_pen_x3.py --window 60 --lead5 --kappa 0.25 --tau 1,
variant B), recalibrated with the same logistic layer fitted on its 2022-2025 out-of-fold
sizing predictions only. The comparison arm PENR_RECAL is the frozen, already scored 2026
prediction (published/penr_2026.parquet, PREREG_PENR.md).

  fit      : python test_pent_2026.py fit --states S --pen P
             (P = dir with penB_w60_l5_k0_t1p0_2022..2025.parquet; reads 2022-2025 outcomes only)
  dry      : python test_pent_2026.py dry --states S --pen P --pen30 PR
             (rehearsal on 2025 vs PENR in PR = penB_w60_l5_k0_2022..2025; no 2026 read)
  predict  : python test_pent_2026.py predict --states S --hf HF --out W
  score    : python test_pent_2026.py score --states S --hf HF --out W   (once; needs Status: FROZEN)

2026 predictions: size_pen_x3.py --season 2026 --variant B --window 60 --lead5 --kappa 0.25 --tau 1
--load-seasons 2021,2022,2023,2024,2025,2026 --part K/4.
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
from test_pen_2026 import N_2026, TERMS, check_inputs, run_tests, sizing_rows, with_r  # noqa: E402

COEF = HERE / "published/pent_recal_coef.json"
PEN_FILE = HERE / "published/penr_2026.parquet"           # frozen PENR_RECAL 2026 predictions (PREREG_PENR)
TEAM_FILE = ROOT / "research/wp_layer/published/pred_2026.parquet"
PENT26_FILES = [HERE / f"published/penB_w60_l5_k0_t1p0_2026_part{k}.parquet" for k in range(4)]
PEN_RESULTS = HERE / "published/results_penr_2026.json"    # its pred_md5 pins PEN_FILE
TAG = "penB_w60_l5_k0_t1p0"
PREREG = HERE / "PREREG_PENT.md"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def frozen(required: bool) -> dict:
    text = PREREG.read_text()
    listed = re.findall(r"^([0-9a-f]{32})  (\S+)$", text, flags=re.M)
    wrong = {p: md5(ROOT / p) for m, p in listed if md5(ROOT / p) != m}
    is_frozen = re.search(r"^Status: FROZEN", text, flags=re.M) is not None
    if required and (not is_frozen or not listed or wrong):
        sys.exit(f"frozen check failed: FROZEN={is_frozen} listed={len(listed)} mismatched={wrong}")
    return {"status_frozen": is_frozen, "n_listed": len(listed), "mismatched": wrong,
            "prereg_md5": md5(PREREG), "script_md5": md5(Path(__file__))}


def w_rows(states: Path, pen: Path, seasons, tag: str | None = None) -> pd.DataFrame:
    """Sizing rows of a layer (files <tag>_<season>.parquet; default TAG)."""
    tmp = {}
    for s in seasons:
        tmp[s] = pd.read_parquet(pen / f"{tag or TAG}_{s}.parquet")
    cols = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings", "home_won"]
    pa = pd.concat([pd.read_parquet(states / f"pa_{s}.parquet", columns=cols) for s in seasons], ignore_index=True)
    d = pd.concat(tmp.values(), ignore_index=True).merge(pa, on=["game_pk", "at_bat_index"], how="left")
    if d[cols].isna().any().any() or len(d) != len(pa):
        sys.exit("sizing rows do not match the plate appearances")
    return with_r(d.rename(columns={"wp_new": "wp_game"}))


def apply(coef: dict, d: pd.DataFrame) -> np.ndarray:
    return 1 / (1 + np.exp(-(design(d, "wp_game", coef["terms"]) @ np.asarray(coef["coef"]))))


def cmd_fit(a) -> None:
    check_inputs(a.states)
    d = w_rows(a.states, a.pen, [2022, 2023, 2024, 2025])
    w = fit(design(d, "wp_game", TERMS), d.home_won.to_numpy(float))
    out = {"base": "wp_game (size_pen_x3.py variant B, --window 60 --lead5 --kappa 0.25 --tau 1)", "terms": TERMS, "coef": [float(x) for x in w],
           "fitted_on": [2022, 2023, 2024, 2025], "n": int(len(d)),
           "pen_md5": {s: md5(a.pen / f"{TAG}_{s}.parquet") for s in (2022, 2023, 2024, 2025)}}
    COEF.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_dry(a) -> None:
    """Rehearsal of the comparison on 2025 (sizing rows only, no 2026 read)."""
    check_inputs(a.states)
    tr, te = w_rows(a.states, a.pen, [2022, 2023, 2024]), w_rows(a.states, a.pen, [2025])
    tr0, te0 = w_rows(a.states, a.pen30, [2022, 2023, 2024], "penB_w60_l5_k0"), w_rows(a.states, a.pen30, [2025], "penB_w60_l5_k0")
    assert (te[["game_pk", "at_bat_index"]].to_numpy() == te0[["game_pk", "at_bat_index"]].to_numpy()).all()
    pw = apply({"terms": TERMS, "coef": fit(design(tr, "wp_game", TERMS), tr.home_won.to_numpy(float))}, te)
    p0 = apply({"terms": TERMS, "coef": fit(design(tr0, "wp_game", TERMS), tr0.home_won.to_numpy(float))}, te0)
    y = te.home_won.to_numpy(float)
    se = {"PENT_RECAL": (pw - y) ** 2, "PENR_RECAL": (p0 - y) ** 2}
    tests, holm, ng = run_w_tests(te.game_pk.to_numpy(), se, (("PENT_RECAL_minus_PENR_RECAL", "PENR_RECAL"),))
    print(json.dumps({"n_rows": len(te), "n_games": ng, "brier": {k: float(v.mean()) for k, v in se.items()},
                      "tests": tests, "holm": holm}, indent=1))


def run_w_tests(g, se, pairs):
    """test_pen_2026.run_tests with PENT_RECAL as the first arm."""
    se2 = {("PEN_RECAL" if k == "PENT_RECAL" else "_" + k): v for k, v in se.items()}   # run_tests' first-arm key
    pairs2 = tuple((h, "_" + o) for h, o in pairs)
    return run_tests(g, se2, pairs2)


def pent_2026_rows() -> pd.DataFrame:
    coef = json.loads(COEF.read_text())
    w = pd.concat([pd.read_parquet(f) for f in PENT26_FILES], ignore_index=True)
    base = pd.read_parquet(PEN_FILE)
    d = base.merge(w[["game_pk", "at_bat_index", "wp_new"]], on=["game_pk", "at_bat_index"], how="left")
    if len(w) != N_2026 or len(d) != N_2026 or d.wp_new.isna().any() or w.duplicated(["game_pk", "at_bat_index"]).any():
        sys.exit("2026 PENT predictions incomplete")
    cols = ["game_pk", "at_bat_index", "inning", "half", "outs", "scheduled_innings"]
    pa = pd.read_parquet(ROOT_STATES / "pa_2026.parquet", columns=cols)    # state columns only, no outcome
    d = with_r(d.merge(pa, on=["game_pk", "at_bat_index"], how="left").rename(columns={"wp_new": "wp_game"}))
    res = d[["game_pk", "at_bat_index", "p_PENR_RECAL"]].copy()
    res["wp_pent"] = d.wp_game
    res["p_PENT_RECAL"] = apply(coef, d)
    if not np.isfinite(res.drop(columns=["game_pk", "at_bat_index"]).to_numpy()).all():
        sys.exit("non-finite predictions")
    return res


ROOT_STATES = Path(".")


def cmd_predict(a) -> None:
    global ROOT_STATES
    ROOT_STATES = a.states
    check_inputs(a.states, a.hf)
    res = pent_2026_rows()
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out / "pent_2026.parquet", index=False)
    print("predictions", len(res), {f.name: md5(f) for f in PENT26_FILES})


def cmd_score(a) -> None:
    global ROOT_STATES
    ROOT_STATES = a.states
    check_inputs(a.states, a.hf)
    res = {"frozen": frozen(required=True),
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    d = pd.read_parquet(a.out / "pent_2026.parquet")
    res["pred_md5"] = md5(a.out / "pent_2026.parquet")
    res["part_md5"] = {f.name: md5(f) for f in PENT26_FILES}
    if md5(PEN_FILE) != json.loads(PEN_RESULTS.read_text())["pred_md5"]:
        sys.exit("penr_2026.parquet is not the file scored in PREREG_PENR")
    again = pent_2026_rows()          # the predictions must come from the frozen parts and coefficients
    if not (again[["game_pk", "at_bat_index"]].equals(d[["game_pk", "at_bat_index"]])
            and np.array_equal(again.p_PENT_RECAL.to_numpy(), d.p_PENT_RECAL.to_numpy())
            and np.array_equal(again.p_PENR_RECAL.to_numpy(), d.p_PENR_RECAL.to_numpy())
            and np.array_equal(again.wp_pent.to_numpy(), d.wp_pent.to_numpy())):
        sys.exit("pent_2026.parquet does not equal the frozen parts + coefficients")
    d = d.merge(pd.read_parquet(TEAM_FILE, columns=["game_pk", "at_bat_index", "p_TEAM"]), on=["game_pk", "at_bat_index"], how="left")
    y_tab = pd.read_parquet(a.states / "pa_2026.parquet", columns=["game_pk", "at_bat_index", "home_won"])
    d = d.merge(y_tab, on=["game_pk", "at_bat_index"], how="left")
    if len(d) != N_2026 or d[["p_TEAM", "home_won"]].isna().any().any():
        sys.exit("gate failed")
    y = d.home_won.to_numpy(float)
    arms = {"PENT_RECAL": d.p_PENT_RECAL, "PENR_RECAL": d.p_PENR_RECAL, "TEAM": d.p_TEAM, "PENT": d.wp_pent}
    se = {k: (v.to_numpy() - y) ** 2 for k, v in arms.items()}
    res["arms"] = {}
    for k, v in arms.items():
        v = v.to_numpy(); p = np.clip(v, 1e-4, 1 - 1e-4)
        band = (v >= 0.3) & (v < 0.5)
        res["arms"][k] = {"brier": float(se[k].mean()), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                          "band_0.3_0.5_gap_pt": float(100 * (v[band].mean() - y[band].mean()))}
    tests, holm, ng = run_w_tests(d.game_pk.to_numpy(), se, (("H1_PENT_RECAL_minus_PENR_RECAL", "PENR_RECAL"),
                                                            ("H2_PENT_RECAL_minus_TEAM", "TEAM")))
    res.update({"n_rows": int(len(d)), "n_games": ng, "tests": tests, "holm_adjusted_p": holm,
                "supported": {h: bool(v <= 0.05) for h, v in holm.items()}})
    (a.out / "results_pent_2026.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "dry", "predict", "score"])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--pen", type=Path, help="size_pen_x3.py --window 60 --lead5 --kappa 0.25 --tau 1 sizing output (penB_w60_l5_k0_t1p0_2022..2025)")
    ap.add_argument("--pen30", type=Path, help="size_pen_x2.py --window 60 --lead5 --kappa 0.25 sizing output (penB_w60_l5_k0_2022..2025) for dry")
    ap.add_argument("--hf", type=Path, help="dir with HF park_factors.parquet (md5-checked)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    {"fit": cmd_fit, "dry": cmd_dry, "predict": cmd_predict, "score": cmd_score}[a.mode](a)


if __name__ == "__main__":
    main()
