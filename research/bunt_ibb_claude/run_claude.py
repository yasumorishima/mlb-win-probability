"""
Call Claude for the bunt / walk-candidate study (PREREG.md). The key is
passed only to this command, e.g.

  env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY="$MLBWP_API_KEY" \
      python research/bunt_ibb_claude/run_claude.py pilot --work W

  pilot:   ... pilot --work W                     # 40 live calls (both parts, low and medium)
  submit:  ... submit --work W --effort E --arm B --splits test,control
  repeat:  ... submit --work W --effort E --arm B --splits test --repeat 100
  retry:   ... submit --work W --effort E --arm B --splits test,control --retry-of B_test-control_E
  collect: ... collect --work W --batch <id> --tag <tag>

Every response is written to W/raw/<tag>.jsonl BEFORE it is parsed:
custom_id, arm, part, model that answered, stop_reason, usage, raw text.
Submitting refuses to exceed the cost cap of PREREG.md (USD 45, batch price,
counted over every batch in W/batches.jsonl).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402

PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6     # claude-opus-5-5, USD per token (live)
SEED = 20261026
CAP_USD = 45.0
EST_TOKENS = {"in": 1300, "out": 250}           # per call, used only for the cap check before a pilot exists


def load_payloads(work: Path, arm: str) -> list[dict]:
    with open(work / f"payload_{arm}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _split_id(custom_id: str) -> tuple[str, str]:
    arm, base = custom_id.split("-", 1)
    if arm not in ("B", "C"):
        sys.exit(f"custom_id without an arm prefix: {custom_id}")
    return arm, base


def _record(msg, custom_id, arm, effort) -> dict:
    text = next((b.text for b in msg.content if b.type == "text"), None)
    return {"custom_id": custom_id, "arm": arm, "part": custom_id.split("-", 1)[0], "effort": effort,
            "model": msg.model, "stop_reason": msg.stop_reason,
            "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens,
                      "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                      "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0},
            "text": text}


def _cost(u: dict, batch: bool) -> float:
    c = (u["input"] + u.get("cache_write", 0) * 1.25 + u.get("cache_read", 0) * 0.1) * PRICE_IN \
        + u["output"] * PRICE_OUT
    return 0.5 * c if batch else c


def _per_call(work: Path, effort: str) -> float:
    p = work / "raw" / "pilot.jsonl"
    rs = [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []
    rs = [r for r in rs if r["effort"] == effort and r.get("usage")]
    if not rs:
        return 0.5 * (EST_TOKENS["in"] * PRICE_IN + EST_TOKENS["out"] * PRICE_OUT)
    return sum(_cost(r["usage"], batch=True) for r in rs) / len(rs)


def pilot(work: Path) -> None:
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    if (raw / "pilot.jsonl").exists():
        sys.exit("pilot already run (raw/pilot.jsonl exists)")
    B, C = load_payloads(work, "B"), load_payloads(work, "C")
    with open(raw / "pilot.jsonl", "a", encoding="utf-8") as out:
        for part in ("bunt", "obp"):
            b = [p for p in B if p["split"] == "pilot" and p["part"] == part]
            c = [p for p in C if p["split"] == "pilot" and p["part"] == part]
            for i, (pb, pc) in enumerate(zip(b, c)):
                effort = "low" if i < 10 else "medium"
                item, arm = (pb, "B") if i % 2 == 0 else (pc, "C")
                msg = client.messages.create(**prompts.request_params(part, item["payload"], effort))
                rec = _record(msg, item["custom_id"], arm, effort)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                ok = prompts.parse_text(rec["text"], rec["stop_reason"]) is not None
                print(part, effort, arm, item["custom_id"], msg.model, msg.stop_reason, rec["usage"],
                      "parsed" if ok else "PARSE FAIL")
    summarize_pilot(work)


def planned_calls(work: Path) -> int:
    n = sum(1 for p in load_payloads(work, "B") if p["split"] in ("test", "control"))
    return 2 * n + 100


def summarize_pilot(work: Path) -> None:
    recs = [json.loads(l) for l in open(work / "raw" / "pilot.jsonl", encoding="utf-8")]
    n_calls = planned_calls(work)
    for effort in ("low", "medium"):
        rs = [r for r in recs if r["effort"] == effort]
        if not rs:
            continue
        fails = sum(prompts.parse_text(r["text"], r["stop_reason"]) is None for r in rs)
        per = _per_call(work, effort)
        print(f"{effort}: n={len(rs)} parse failures {fails}; mean in "
              f"{sum(r['usage']['input'] for r in rs)/len(rs):.0f} out "
              f"{sum(r['usage']['output'] for r in rs)/len(rs):.0f} tokens; "
              f"batch projection ${per * n_calls:.2f} for {n_calls} calls")
    per_m = _per_call(work, "medium")
    print("rule (PREREG): effort =", "medium" if per_m * n_calls <= 40.0 else "low")


def _spent(work: Path) -> float:
    """Batch cost of everything collected so far plus a projection for
    submitted but not yet collected batches."""
    total = 0.0
    bj = work / "batches.jsonl"
    if not bj.exists():
        return 0.0
    for line in open(bj, encoding="utf-8"):
        b = json.loads(line)
        f = work / "raw" / f"{b['tag']}.jsonl"
        if f.exists():
            total += sum(_cost(r["usage"], batch=True) for r in map(json.loads, open(f, encoding="utf-8"))
                         if r.get("usage"))
        else:
            total += b["n"] * _per_call(work, b["effort"])
    return total


def submit(work: Path, effort: str, arm: str, splits: list[str], repeat: int | None,
           retry_of: str | None = None) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if arm not in ("B", "C"):
        sys.exit("arm must be B or C")
    items = [p for p in load_payloads(work, arm) if p["split"] in splits]
    tag = f"{arm}_{'-'.join(splits)}_{effort}"
    if retry_of:
        if retry_of.split("_")[0] != arm:
            sys.exit(f"--retry-of {retry_of} is not arm {arm}")
        good = set()
        for line in open(work / "raw" / f"{retry_of}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") != arm:
                sys.exit(f"record of arm {r.get('arm')} in {retry_of}")
            if prompts.parse_text(r.get("text"), r.get("stop_reason")) is not None:
                good.add(r["custom_id"])
        items = [p for p in items if p["custom_id"] not in good]
        tag = f"{retry_of}_retry"
        if not items:
            print("nothing to retry")
            return
    if repeat:
        random.Random(SEED).shuffle(items)
        items = items[:repeat]
        tag += f"_repeat{repeat}"
    bj = work / "batches.jsonl"
    if bj.exists() and any(json.loads(l)["tag"] == tag for l in open(bj, encoding="utf-8")):
        sys.exit(f"a batch with tag {tag} was already submitted")
    projected = _spent(work) + len(items) * _per_call(work, effort)
    if projected > CAP_USD:
        sys.exit(f"cost cap: projected ${projected:.2f} > ${CAP_USD}")
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    reqs = [Request(custom_id=f"{arm}-{p['custom_id']}",
                    params=MessageCreateParamsNonStreaming(**prompts.request_params(p["part"], p["payload"], effort)))
            for p in items]
    batch = client.messages.batches.create(requests=reqs)
    rec = {"batch_id": batch.id, "tag": tag, "effort": effort, "n": len(reqs),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "projected_total_usd": round(projected, 2)}
    with open(bj, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def collect(work: Path, batch_id: str, tag: str) -> None:
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    known = {json.loads(l)["batch_id"]: json.loads(l) for l in open(work / "batches.jsonl", encoding="utf-8")}
    if known.get(batch_id, {}).get("tag") != tag:
        sys.exit(f"batch {batch_id} was submitted as {known.get(batch_id, {}).get('tag')!r}, not {tag!r}")
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print("not ended:", b.processing_status, b.request_counts)
        return
    arm, effort = tag.split("_")[0], known[batch_id]["effort"]
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    n = 0
    with open(raw / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            rid_arm, base = _split_id(res.custom_id)
            if rid_arm != arm:
                sys.exit(f"result {res.custom_id} belongs to arm {rid_arm}, batch tag says {arm}")
            if res.result.type == "succeeded":
                rec = _record(res.result.message, base, arm, effort)
            else:
                rec = {"custom_id": base, "arm": arm, "part": base.split("-", 1)[0], "effort": effort,
                       "error": res.result.type}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(tag, "records", n, b.request_counts, f"spent so far ${_spent(work):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "pilot-summary", "submit", "collect"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--effort", default="medium", choices=["low", "medium"])
    ap.add_argument("--arm", default="B")
    ap.add_argument("--splits", default="test,control")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--batch")
    ap.add_argument("--tag")
    ap.add_argument("--retry-of")
    a = ap.parse_args()
    if a.mode == "pilot":
        pilot(a.work)
    elif a.mode == "pilot-summary":
        summarize_pilot(a.work)
    elif a.mode == "submit":
        submit(a.work, a.effort, a.arm, a.splits.split(","), a.repeat, a.retry_of)
    else:
        collect(a.work, a.batch, a.tag)


if __name__ == "__main__":
    main()
