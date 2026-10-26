"""
Pre-registered analysis of the bunt / walk-candidate study (PREREG.md).
Frozen with it.

  python research/bunt_ibb_claude/analyze.py --work W --effort E

Reads W/rows_{bunt,obp}.parquet and W/raw/{B,C}_test-control_<E>.jsonl
(plus *_retry.jsonl and B_test_<E>_repeat100.jsonl). Writes
W/results.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402

SEED = 20261026
N_BOOT = 10000
PARTS = ("bunt", "obp")
FEW = {"bunt": 3, "obp": 100}     # "little history": fewer earlier bunts / PAs than this
PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6


def parse_records(path: Path, arm: str) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != arm:
            sys.exit(f"{path.name}: record of arm {r.get('arm')!r} where {arm} was expected")
        out[r["custom_id"]] = {"p": prompts.parse_text(r.get("text"), r.get("stop_reason")),
                               "model": r.get("model"), "usage": r.get("usage")}
    return out


def load_arm(work: Path, arm: str, effort: str) -> dict:
    base = parse_records(work / "raw" / f"{arm}_test-control_{effort}.jsonl", arm)
    if not base:
        sys.exit(f"no results for arm {arm}")
    retry = parse_records(work / "raw" / f"{arm}_test-control_{effort}_retry.jsonl", arm)
    for k, v in retry.items():
        if base.get(k, {}).get("p") is None:
            base[k] = v
    return base


def ll(p, y):
    p = np.clip(p, 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def weights(n_games: int, rng) -> np.ndarray:
    return rng.multinomial(n_games, np.full(n_games, 1 / n_games), size=N_BOOT)


def boot_mean(d: np.ndarray, games: np.ndarray, rng) -> np.ndarray:
    """Bootstrap over games of mean(d)."""
    ug, gi = np.unique(games, return_inverse=True)
    s = np.bincount(gi, weights=d)
    c = np.bincount(gi).astype(float)
    w = weights(len(ug), rng)
    return (w @ s) / (w @ c)


def auc(p, y):
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib(p, y):
    b = np.clip((p * 10).astype(int), 0, 9)
    return [{"bin": f"{k/10:.1f}", "n": int((b == k).sum()),
             "pred": float(p[b == k].mean()) if (b == k).any() else None,
             "obs": float(y[b == k].mean()) if (b == k).any() else None} for k in range(10)]


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    out, prev = {}, 0.0
    for i, h in enumerate(order):
        adj = max(prev, min(1.0, (len(order) - i) * pvals[h]))
        out[h] = adj
        prev = adj
    return out


def ci(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyze(work: Path, effort: str) -> dict:
    rows = pd.concat([pd.read_parquet(work / f"rows_{p}.parquet") for p in PARTS], ignore_index=True)
    rows = rows[rows.split.isin(["test", "control"])].copy()
    rows["date"] = pd.to_datetime(rows.date)
    arms = {arm: load_arm(work, arm, effort) for arm in ("B", "C")}
    res = {"effort": effort, "n": {}}

    for arm, recs in arms.items():
        p = rows.custom_id.map(lambda k: (recs.get(k) or {}).get("p"))
        for part in PARTS:
            for split in ("test", "control"):
                m = (rows.part == part) & (rows.split == split)
                k = int(p[m].isna().sum())
                res["n"][f"{arm}_{part}_{split}_filled_with_A"] = k
                res["n"][f"{arm}_{part}_{split}_fill_above_2pct"] = bool(k > 0.02 * m.sum())
        rows[f"p_{arm}"] = p.astype(float).fillna(rows.p_model)
        models = {(recs.get(k) or {}).get("model") for k in rows.custom_id}
        res[f"{arm}_answering_models"] = sorted(m for m in models if m)
        use = [v["usage"] for v in recs.values() if v.get("usage")]
        res[f"{arm}_tokens"] = {"input": int(sum(u["input"] for u in use)),
                                "output": int(sum(u["output"] for u in use)),
                                "batch_cost_usd_uncached": round(0.5 * sum(u["input"] * PRICE_IN + u["output"] * PRICE_OUT for u in use), 2)}
    rows["p_A"] = np.where(rows.split == "control", rows.p_a_ctrl, rows.p_a)
    rows["p_L"] = rows.lg_rate

    rng = np.random.default_rng(SEED)
    by = {}
    pvals = {}
    for part in PARTS:
        by[part] = {}
        for split in ("test", "control"):
            d = rows[(rows.part == part) & (rows.split == split)]
            y = d.y.to_numpy().astype(float)
            g = d.game_pk.to_numpy()
            out = {"n": int(len(d)), "games": int(len(np.unique(g))), "y_mean": float(y.mean())}
            for arm in ("A", "B", "C", "L"):
                p = np.clip(d[f"p_{arm}"].to_numpy(), 0.01, 0.99)
                out[arm] = {"log_loss": float(ll(p, y).mean()), "brier": float(((p - y) ** 2).mean()),
                            "auc": auc(p, y), "mean_p": float(p.mean()), "calibration": calib(p, y)}
            for arm in ("B", "C"):
                diff = ll(d[f"p_{arm}"].to_numpy(), y) - ll(d.p_A.to_numpy(), y)
                bs = boot_mean(diff, g, rng)
                out[f"{arm}_minus_A"] = {"mean": float(diff.mean()), "ci95": ci(bs),
                                         "p_one_sided": float((bs >= 0).mean())}
            by[part][split] = out
        pvals[f"H1_{part}_C"] = by[part]["test"]["C_minus_A"]["p_one_sided"]
        pvals[f"H2_{part}_B"] = by[part]["test"]["B_minus_A"]["p_one_sided"]
    res["holm_adjusted_p"] = holm(pvals)
    res["supported"] = {h: bool(v < 0.05) for h, v in res["holm_adjusted_p"].items()}

    # memory check: D = LL(A) - LL(arm); flag if D_control - D_test has a 95% CI above 0
    res["memory_check"], res["aug_sep"], res["skill_claimed"] = {}, {}, {}
    for part in PARTS:
        res["memory_check"][part] = {}
        for arm in ("B", "C"):
            bs = {}
            for split in ("test", "control"):
                d = rows[(rows.part == part) & (rows.split == split)]
                y = d.y.to_numpy().astype(float)
                bs[split] = boot_mean(ll(d.p_A.to_numpy(), y) - ll(d[f"p_{arm}"].to_numpy(), y), d.game_pk.to_numpy(), rng)
            gap = bs["control"] - bs["test"]
            res["memory_check"][part][arm] = {"ci95": ci(gap), "consistent_with_memorization": bool(np.percentile(gap, 2.5) > 0)}
        late = rows[(rows.part == part) & (rows.split == "test") & (rows.date >= "2026-08-01")]
        yl = late.y.to_numpy().astype(float)
        a = {"n": int(len(late))}
        for arm in ("A", "B", "C"):
            a[arm] = float(ll(late[f"p_{arm}"].to_numpy(), yl).mean())
        for arm in ("B", "C"):
            dl = ll(late[f"p_{arm}"].to_numpy(), yl) - ll(late.p_A.to_numpy(), yl)
            a[f"{arm}_minus_A"] = {"mean": float(dl.mean()), "ci95": ci(boot_mean(dl, late.game_pk.to_numpy(), rng))}
        res["aug_sep"][part] = a
        for h, arm in ((f"H1_{part}_C", "C"), (f"H2_{part}_B", "B")):
            ok = res["supported"][h]
            if ok and res["memory_check"][part][arm]["consistent_with_memorization"]:
                ok = a[f"{arm}_minus_A"]["ci95"][1] < 0
            res["skill_claimed"][h] = bool(ok)

    # reported regardless of outcome
    res["few_history"], res["corr_C_adjustment_vs_residual"] = {}, {}
    for part in PARTS:
        t = rows[(rows.part == part) & (rows.split == "test")]
        y = t.y.to_numpy().astype(float)
        few = t[t.bat_n_before < FEW[part]]
        res["few_history"][part] = {"threshold": FEW[part], "n": int(len(few))} | {
            arm: float(ll(few[f"p_{arm}"].to_numpy(), few.y.to_numpy().astype(float)).mean()) if len(few) else None
            for arm in ("A", "B", "C")}
        adj, resid = (t.p_C - t.p_A).to_numpy(), y - t.p_A.to_numpy()
        res["corr_C_adjustment_vs_residual"][part] = float(np.corrcoef(adj, resid)[0, 1]) if adj.std() > 0 else None
        res.setdefault("mean_abs_C_minus_A", {})[part] = float(np.abs(adj).mean())

    rep = parse_records(work / "raw" / f"B_test_{effort}_repeat100.jsonl", "B")
    pairs = [(v["p"], arms["B"].get(k, {}).get("p")) for k, v in rep.items()]
    pairs = [(x, z) for x, z in pairs if x is not None and z is not None]
    if pairs:
        dd = np.array([x - z for x, z in pairs])
        res["repeat_B"] = {"n": len(pairs), "sd_of_difference": float(dd.std()),
                           "mean_abs_difference": float(np.abs(dd).mean())}
    res["by_part"] = by
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--effort", required=True)
    a = ap.parse_args()
    res = analyze(a.work, a.effort)
    (a.work / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "holm_adjusted_p", "supported", "memory_check", "aug_sep",
                                         "skill_claimed")}, indent=1))
    for part in PARTS:
        for split in ("test", "control"):
            o = res["by_part"][part][split]
            print(part, split, {arm: round(o[arm]["log_loss"], 4) for arm in ("A", "B", "C", "L")},
                  "C-A", round(o["C_minus_A"]["mean"], 5), [round(x, 5) for x in o["C_minus_A"]["ci95"]],
                  "B-A", round(o["B_minus_A"]["mean"], 5), [round(x, 5) for x in o["B_minus_A"]["ci95"]])


if __name__ == "__main__":
    main()
