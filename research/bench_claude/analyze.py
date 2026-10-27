"""
Scores the bench-call study once (PREREG.md):

  python analyze.py --inputs research/bench_claude/inputs --work W --out research/bench_claude/published/results.json

Agreement = the arm's pick is the player the manager actually sent in.
Arms: C (Claude with the model's numbers), B (Claude alone), MODEL (the
player layer's best candidate), USAGE (pitching changes: most-used available
reliever in that inning and lead state over the previous 30 days).
An answer that cannot be parsed, or names a player who is not a candidate,
counts as a miss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED, NBOOT = 20261028, 10_000
PRICE = {"input": 4.0, "cache_write": 5.0, "cache_read": 0.4, "output": 20.0}    # $/MTok, claude-opus-5-5; batch = half


def parse(rec: dict | None, cands: list[int]) -> tuple[int | None, list[int]]:
    if not rec or rec.get("error") or rec.get("stop_reason") in ("refusal", "max_tokens") or not rec.get("text"):
        return None, []
    try:
        x = json.loads(rec["text"])
        choice = int(x["choice"])
        top = [int(v) for v in x["top3"]]
    except (ValueError, KeyError, TypeError):
        return None, []
    top3 = []
    for v in [choice] + top:
        if v in cands and v not in top3:
            top3.append(v)
    return (choice if choice in cands else None), top3[:3]


def boot_diff(a: np.ndarray, b: np.ndarray, games: np.ndarray, rng) -> dict:
    d = a.astype(float) - b.astype(float)
    ug = np.unique(games)
    idx = {g: np.flatnonzero(games == g) for g in ug}
    sums = np.array([d[idx[g]].sum() for g in ug]); ns = np.array([len(idx[g]) for g in ug])
    bs = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = rng.integers(0, len(ug), len(ug))
        bs[i] = sums[pick].sum() / ns[pick].sum()
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(p)}


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = run
    return out


def cost(recs: list[dict]) -> float:
    tot = 0.0
    for r in recs:
        u = r.get("usage")
        if u:
            tot += (u["input"] * PRICE["input"] + u["cache_write"] * PRICE["cache_write"]
                    + u["cache_read"] * PRICE["cache_read"] + u["output"] * PRICE["output"]) / 1e6
    return tot / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.inputs / "items_test.jsonl", encoding="utf-8")]
    raw = {}
    recs = {}
    for arm in ("B", "C"):
        recs[arm] = [json.loads(l) for l in open(a.work / f"raw_test_{arm}.jsonl", encoding="utf-8")]
        raw[arm] = {r["custom_id"].rsplit("|", 1)[0]: r for r in recs[arm]}
    res = {"n_items": len(items), "excluded": {}, "cost_usd_batch": {arm: round(cost(recs[arm]), 2) for arm in recs}}
    for it in items:
        if it.get("excluded"):
            res["excluded"][it["excluded"]] = res["excluded"].get(it["excluded"], 0) + 1
    rng = np.random.default_rng(SEED)
    for task in ("pitching_change", "pinch_hitter"):
        T = [it for it in items if it["task"] == task and not it.get("excluded")]
        g = np.array([it["game_pk"] for it in T])
        act = [it["actual"] for it in T]
        agree, top3, invalid = {}, {}, {}
        for arm in ("B", "C"):
            picks = [parse(raw[arm].get(it["id"]), it["candidates"]) for it in T]
            agree[arm] = np.array([p[0] == y for p, y in zip(picks, act)])
            top3[arm] = np.array([y in p[1] for p, y in zip(picks, act)])
            invalid[arm] = int(sum(1 for p in picks if p[0] is None))
        agree["MODEL"] = np.array([it["model_pick"] == y for it, y in zip(T, act)])
        mtop = [sorted(it["model_value"], key=lambda k: -it["model_value"][k])[:3] for it in T]
        top3["MODEL"] = np.array([str(y) in m for m, y in zip(mtop, act)])
        if task == "pitching_change":
            agree["USAGE"] = np.array([it["usage_pick"] == y for it, y in zip(T, act)])
        n_c = [len(it["candidates"]) for it in T]
        r = {"n": len(T), "games": int(len(np.unique(g))), "mean_candidates": float(np.mean(n_c)) if T else None,
             "chance_top1": float(np.mean([1 / c for c in n_c])) if T else None,
             "agreement": {k: float(v.mean()) for k, v in agree.items()},
             "top3_agreement": {k: float(v.mean()) for k, v in top3.items()},
             "invalid_answers": invalid}
        if len(T) == 0:
            res[task] = r
            continue
        comps = [("C", "MODEL"), ("B", "MODEL"), ("C", "B")] + ([("C", "USAGE"), ("B", "USAGE")] if task == "pitching_change" else [])
        r["differences"] = {f"{x}-{y}": boot_diff(agree[x], agree[y], g, rng) for x, y in comps}
        if task == "pitching_change":
            late = np.array([it["inning"] >= 8 for it in T])
            r["by_inning"] = {lab: {k: float(v[mask].mean()) for k, v in agree.items()} | {"n": int(mask.sum())}
                              for lab, mask in (("6-7", ~late), ("8+", late))}
        res[task] = r
    pc = res["pitching_change"]["differences"]
    adj = holm({"H1": pc["C-MODEL"]["p"], "H2": pc["C-USAGE"]["p"]})
    res["hypotheses"] = {
        "H1 C > MODEL (pitching changes)": {**pc["C-MODEL"], "p_holm": adj["H1"],
                                            "supported": bool(adj["H1"] < 0.05 and pc["C-MODEL"]["diff"] > 0)},
        "H2 C > USAGE (pitching changes)": {**pc["C-USAGE"], "p_holm": adj["H2"],
                                            "supported": bool(adj["H2"] < 0.05 and pc["C-USAGE"]["diff"] > 0)},
    }
    res["answering_models"] = sorted({r.get("model") for arm in recs for r in recs[arm] if r.get("model")})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
