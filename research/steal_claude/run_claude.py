"""
Call Claude for the steal study (run where ANTHROPIC_API_KEY is available).

  pilot:   python run_claude.py pilot --work W            # live calls, 20 per effort level
  submit:  python run_claude.py submit --work W --effort E --arm B --splits test,control
  repeat:  python run_claude.py submit --work W --effort E --arm B --splits test --repeat 100
  retry:   python run_claude.py submit --work W --effort E --arm B --splits test,control --retry-of B_test-control_E
  collect: python run_claude.py collect --work W --batch <id> --tag <name>

Every response is written to W/raw/<tag>.jsonl BEFORE it is parsed:
custom_id, arm, model that answered, stop_reason, usage, raw text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import strategist  # noqa: E402

PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6     # claude-opus-5-5, USD per token (live)
SEED = 20261015


def load_payloads(work: Path, arm: str) -> list[dict]:
    with open(work / f"payload_{arm}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _split_id(custom_id: str) -> tuple[str, str]:
    """Request ids carry the arm: "B-<row id>" / "C-<row id>"."""
    arm, base = custom_id.split("-", 1)
    if arm not in ("B", "C"):
        sys.exit(f"custom_id without an arm prefix: {custom_id}")
    return arm, base


def _record(msg, custom_id, arm, effort) -> dict:
    text = next((b.text for b in msg.content if b.type == "text"), None)
    return {"custom_id": custom_id, "arm": arm, "effort": effort, "model": msg.model,
            "stop_reason": msg.stop_reason,
            "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens},
            "text": text}


def pilot(work: Path) -> None:
    import anthropic
    client = anthropic.Anthropic()
    rows = {arm: [p for p in load_payloads(work, arm) if p["split"] == "pilot"] for arm in ("B", "C")}
    raw = work / "raw"
    raw.mkdir(exist_ok=True)
    out = open(raw / "pilot.jsonl", "a", encoding="utf-8")
    for effort, sl in (("low", slice(0, 20)), ("medium", slice(20, 40))):
        for i, (b, c) in enumerate(zip(rows["B"][sl], rows["C"][sl])):
            item, arm = (b, "B") if i % 2 == 0 else (c, "C")
            params = strategist.request_params("steal", item["payload"], effort=effort)
            msg = client.messages.create(**params)
            rec = _record(msg, item["custom_id"], arm, effort)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            ok = strategist.parse_response("steal", msg) is not None
            print(effort, arm, item["custom_id"], msg.stop_reason, rec["usage"], "parsed" if ok else "PARSE FAIL")
    out.close()
    summarize_pilot(work)


def summarize_pilot(work: Path) -> None:
    recs = [json.loads(l) for l in open(work / "raw" / "pilot.jsonl", encoding="utf-8")]
    n_calls = 2 * sum(1 for p in load_payloads(work, "B") if p["split"] in ("test", "control")) + 100
    for effort in ("low", "medium"):
        rs = [r for r in recs if r["effort"] == effort]
        if not rs:
            continue
        live = sum(r["usage"]["input"] * PRICE_IN + r["usage"]["output"] * PRICE_OUT for r in rs) / len(rs)
        print(f"{effort}: n={len(rs)} mean in {sum(r['usage']['input'] for r in rs)/len(rs):.0f} "
              f"out {sum(r['usage']['output'] for r in rs)/len(rs):.0f} tokens, "
              f"live ${live:.4f}/call, batch projection ${0.5*live*n_calls:.2f} for {n_calls} calls")


def submit(work: Path, effort: str, arm: str, splits: list[str], repeat: int | None,
           retry_of: str | None = None) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    items = [p for p in load_payloads(work, arm) if p["split"] in splits]
    tag = f"{arm}_{'-'.join(splits)}_{effort}"
    if retry_of:
        # resubmit, once, the requests of an earlier batch that did not parse
        good = set()
        if retry_of.split("_")[0] != arm:
            sys.exit(f"--retry-of {retry_of} is not arm {arm}")
        for line in open(work / "raw" / f"{retry_of}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm") != arm:
                sys.exit(f"record of arm {r.get('arm')} in {retry_of}")
            if r.get("text") and r.get("stop_reason") == "end_turn":
                try:
                    est = strategist.StealEstimate.model_validate_json(r["text"])
                    if 0.0 <= est.p_success <= 1.0:
                        good.add(r["custom_id"])
                except Exception:
                    pass
        items = [p for p in items if p["custom_id"] not in good]
        tag = f"{retry_of}_retry"
        if not items:
            print("nothing to retry")
            return
    if repeat:
        random.Random(SEED).shuffle(items)
        items = items[:repeat]
        tag += f"_repeat{repeat}"
    reqs = [Request(custom_id=f"{arm}-{p['custom_id']}",
                    params=MessageCreateParamsNonStreaming(**strategist.request_params(
                        "steal", p["payload"], effort=effort)))
            for p in items]
    batch = client.messages.batches.create(requests=reqs)
    rec = {"batch_id": batch.id, "tag": tag, "n": len(reqs), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(work / "batches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def collect(work: Path, batch_id: str, tag: str) -> None:
    import anthropic
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print("not ended:", b.processing_status, b.request_counts)
        return
    arm = tag.split("_")[0]
    effort = tag.split("_")[2]
    known = {json.loads(l)["batch_id"]: json.loads(l)["tag"]
             for l in open(work / "batches.jsonl", encoding="utf-8")}
    if known.get(batch_id) != tag:
        sys.exit(f"batch {batch_id} was submitted as {known.get(batch_id)!r}, not {tag!r}")
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
                rec = {"custom_id": base, "arm": arm, "effort": effort, "error": res.result.type}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(tag, "records", n, b.request_counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "pilot-summary", "submit", "collect"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--arm", default="B")
    ap.add_argument("--splits", default="test,control")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--batch")
    ap.add_argument("--tag")
    ap.add_argument("--retry-of", help="tag of a collected batch whose failures are resubmitted once")
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
