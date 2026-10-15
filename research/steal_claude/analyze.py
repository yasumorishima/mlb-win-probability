"""
Pre-registered analysis of the steal study (PREREG.md). Frozen with it.

  python research/steal_claude/analyze.py --work W --effort E > W/results.txt

Reads W/rows.parquet and W/raw/{B,C}_test-control_<E>.jsonl (plus
*_retry.jsonl for requests that failed the first time and
B_test_<E>_repeat100.jsonl). Writes W/results.json and prints a summary.
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
import strategist  # noqa: E402

SEED = 20261015
N_BOOT = 10000
PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6


def _parse_records(path: Path, arm: str) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != arm:
            sys.exit(f"{path.name}: record of arm {r.get('arm')!r} where {arm} was expected")
        p = None
        if r.get("text") and r.get("stop_reason") == "end_turn":
            try:
                est = strategist.StealEstimate.model_validate_json(r["text"])
                if 0.0 <= est.p_success <= 1.0:
                    p = est.p_success
            except Exception:
                p = None
        out[r["custom_id"]] = {"p": p, "model": r.get("model"), "usage": r.get("usage")}
    return out


def load_arm(work: Path, arm: str, effort: str) -> dict:
    base = _parse_records(work / "raw" / f"{arm}_test-control_{effort}.jsonl", arm)
    if not base:
        sys.exit(f"no results for arm {arm}")
    retry = _parse_records(work / "raw" / f"{arm}_test-control_{effort}_retry.jsonl", arm)
    for k, v in retry.items():
        if base.get(k, {}).get("p") is None:
            base[k] = v
    return base


def ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(d: np.ndarray, games: np.ndarray, rng) -> np.ndarray:
    """Bootstrap over games of mean(d)."""
    ug, gi = np.unique(games, return_inverse=True)
    s = np.bincount(gi, weights=d)
    c = np.bincount(gi).astype(float)
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)
    return (w @ s) / (w @ c)


def auc(p, y):
    ranks = pd.Series(p).rank(method="average").to_numpy()   # midranks for ties
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib(p, y):
    b = np.clip((p * 10).astype(int), 0, 9)
    return [{"bin": f"{k/10:.1f}", "n": int((b == k).sum()),
             "pred": float(p[b == k].mean()) if (b == k).any() else None,
             "obs": float(y[b == k].mean()) if (b == k).any() else None} for k in range(10)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--effort", required=True)
    a = ap.parse_args()
    rows = pd.read_parquet(a.work / "rows.parquet")
    rows = rows[rows.split.isin(["test", "control"])].copy()
    arms = {arm: load_arm(a.work, arm, a.effort) for arm in ("B", "C")}
    res = {"effort": a.effort, "n": {}}

    # fill failures with the model's estimate (counted)
    for arm, recs in arms.items():
        p = rows.custom_id.map(lambda k: (recs.get(k) or {}).get("p"))
        for split in ("test", "control"):
            m = rows.split == split
            k = int(p[m].isna().sum())
            res["n"][f"{arm}_{split}_filled_with_A"] = k
            res["n"][f"{arm}_{split}_fill_above_2pct"] = bool(k > 0.02 * m.sum())
        rows[f"p_{arm}"] = p.fillna(rows.p_model).astype(float)
        models = {(recs.get(k) or {}).get("model") for k in rows.custom_id}
        res[f"{arm}_answering_models"] = sorted(m for m in models if m)
        use = [v["usage"] for v in recs.values() if v.get("usage")]
        res[f"{arm}_tokens"] = {"input": int(sum(u["input"] for u in use)),
                                "output": int(sum(u["output"] for u in use)),
                                "batch_cost_usd": round(0.5 * sum(u["input"] * PRICE_IN + u["output"] * PRICE_OUT for u in use), 2)}
    rows["p_A"] = np.where(rows.split == "control", rows.p_a_ctrl, rows.p_a)
    rows["p_L"] = rows.season_rate_before

    rng = np.random.default_rng(SEED)
    out = {}
    for split in ("test", "control"):
        d = rows[rows.split == split]
        y = d.success.to_numpy().astype(float)
        g = d.game_pk.to_numpy()
        part = {"n": int(len(d)), "games": int(len(np.unique(g)))}
        for arm in ("A", "B", "C", "L"):
            p = d[f"p_{arm}"].to_numpy()
            part[arm] = {"log_loss": float(ll(p, y).mean()), "brier": float(((np.clip(p, .01, .99) - y) ** 2).mean()),
                         "auc": auc(np.clip(p, .01, .99), y), "calibration": calib(np.clip(p, .01, .99), y)}
        for arm in ("B", "C"):
            diff = ll(d[f"p_{arm}"].to_numpy(), y) - ll(d.p_A.to_numpy(), y)
            bs = boot_diff(diff, g, rng)
            part[f"{arm}_minus_A"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      "p_one_sided": float((bs >= 0).mean())}
        out[split] = part

    # H1 (C < A), H2 (B < A) on test, Holm
    pt = {"H1_C": out["test"]["C_minus_A"]["p_one_sided"], "H2_B": out["test"]["B_minus_A"]["p_one_sided"]}
    order = sorted(pt, key=pt.get)
    holm, prev = {}, 0.0
    for i, h in enumerate(order):
        adj = max(prev, min(1.0, (len(order) - i) * pt[h]))
        holm[h] = adj
        prev = adj
    res["holm_adjusted_p"] = holm
    res["supported"] = {h: holm[h] < 0.05 for h in holm}
    res["mde_note"] = "differences smaller than about 0.010 in log loss are not detectable with this test set (PREREG)"

    # memory: D = LL(A) - LL(arm); flag if D_control - D_test has a 95% CI above 0
    mem = {}
    for arm in ("B", "C"):
        bs = {}
        for split in ("test", "control"):
            d = rows[rows.split == split]
            y = d.success.to_numpy().astype(float)
            diff = ll(d.p_A.to_numpy(), y) - ll(d[f"p_{arm}"].to_numpy(), y)
            bs[split] = boot_diff(diff, d.game_pk.to_numpy(), rng)
        gap = bs["control"] - bs["test"]
        mem[arm] = {"ci95": [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))],
                    "consistent_with_memorization": bool(np.percentile(gap, 2.5) > 0)}
    res["memory_check"] = mem

    t = rows[rows.split == "test"].copy()
    y = t.success.to_numpy().astype(float)
    late = t[pd.to_datetime(t.date) >= "2026-08-01"]
    yl = late.success.to_numpy().astype(float)
    res["aug_sep"] = {"n": int(len(late))}
    for arm in ("A", "B", "C"):
        res["aug_sep"][arm] = float(ll(late[f"p_{arm}"].to_numpy(), yl).mean())
    for arm in ("B", "C"):
        dl = ll(late[f"p_{arm}"].to_numpy(), yl) - ll(late.p_A.to_numpy(), yl)
        bl = boot_diff(dl, late.game_pk.to_numpy(), rng)
        res["aug_sep"][f"{arm}_minus_A"] = {"mean": float(dl.mean()),
                                            "ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))]}
    # reading rule (PREREG): a supported hypothesis counts as skill only if the
    # arm is not flagged by the memory check, or its August-September
    # interval also lies below 0
    res["skill_claimed"] = {}
    for h, arm in (("H1_C", "C"), ("H2_B", "B")):
        ok = res["supported"][h]
        if ok and res["memory_check"][arm]["consistent_with_memorization"]:
            ok = res["aug_sep"][f"{arm}_minus_A"]["ci95"][1] < 0
        res["skill_claimed"][h] = bool(ok)
    few = t[t.runner_att_before < 5]
    res["runners_fewer_than_5_attempts"] = {arm: float(ll(few[f"p_{arm}"].to_numpy(), few.success.to_numpy().astype(float)).mean())
                                            for arm in ("A", "B", "C")} | {"n": int(len(few))}
    adj, resid = t.p_C - t.p_A, y - t.p_A
    res["corr_C_adjustment_vs_residual"] = float(np.corrcoef(adj, resid)[0, 1]) if adj.std() > 0 else None

    # secondary (one-sided, alpha .05, not part of Holm): B gains more where
    # previous-season Savant values are missing
    miss = (t.runner_sprint_prev.isna() | t.catcher_pop_prev.isna()).to_numpy()
    gain = ll(t.p_A.to_numpy(), y) - ll(t.p_B.to_numpy(), y)
    g = t.game_pk.to_numpy()
    ug, gi = np.unique(g, return_inverse=True)
    sm = np.bincount(gi, weights=gain * miss, minlength=len(ug))
    cm = np.bincount(gi, weights=miss.astype(float), minlength=len(ug))
    sn = np.bincount(gi, weights=gain * ~miss, minlength=len(ug))
    cn = np.bincount(gi, weights=(~miss).astype(float), minlength=len(ug))
    w = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=N_BOOT)   # same games for both groups
    b_m, b_n = (w @ sm) / (w @ cm), (w @ sn) / (w @ cn)
    res["secondary_missing_savant"] = {
        "n_missing": int(miss.sum()), "gain_missing": float(gain[miss].mean()), "gain_present": float(gain[~miss].mean()),
        "p_one_sided": float((b_m - b_n <= 0).mean()),
        "auc_missing": {arm: auc(t[f"p_{arm}"].to_numpy()[miss], y[miss]) for arm in ("A", "B")}}

    rep = _parse_records(a.work / "raw" / f"B_test_{a.effort}_repeat100.jsonl", "B")
    pairs = [(v["p"], arms["B"].get(k, {}).get("p")) for k, v in rep.items()]
    pairs = [(x, z) for x, z in pairs if x is not None and z is not None]
    if pairs:
        dd = np.array([x - z for x, z in pairs])
        res["repeat_B"] = {"n": len(pairs), "sd_of_difference": float(dd.std()), "mean_abs_difference": float(np.abs(dd).mean())}

    res["by_split"] = out
    (a.work / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "holm_adjusted_p", "supported", "memory_check", "aug_sep",
                                         "skill_claimed")}, indent=1))
    for split in ("test", "control"):
        print(split, {arm: round(out[split][arm]["log_loss"], 4) for arm in ("A", "B", "C", "L")},
              "C-A", out[split]["C_minus_A"]["mean"], out[split]["C_minus_A"]["ci95"],
              "B-A", out[split]["B_minus_A"]["mean"], out[split]["B_minus_A"]["ci95"])


if __name__ == "__main__":
    main()
