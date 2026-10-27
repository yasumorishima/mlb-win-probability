"""
Claude's bench calls (PREREG.md). Arms: B = without the model's numbers,
C = with them (payload_B / payload_C of build_items.py). Same system prompt.

  pilot  : python run_claude.py pilot  --inputs I --work W            # live calls on the pilot items (both arms)
  cost   : python run_claude.py pilot_cost --work W --n-test-calls N  # the PREREG cost cap
  submit : python run_claude.py submit --inputs I --work W --arm B|C [--retry]   # Batches API, test items not excluded
  collect: python run_claude.py collect --work W --arm B|C [--retry]

Raw answers are appended to W/raw_<set>_<arm>.jsonl (custom_id, model, stop_reason,
usage, text). The key is read from ANTHROPIC_API_KEY only, which is set for
the one command that calls the API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
SYSTEM = (
    "You are the bench coach of a Major League Baseball team. You get one game moment as JSON: "
    "date, teams, inning, half, outs, runners, score, the decision your manager has already made "
    "(change pitchers now, or send a pinch hitter now) and the players available for it, with their "
    "records. Choose the one player you would send in. Use your baseball judgment: the hitters or pitcher "
    "you will face, handedness, the inning and score, how the team has used its relievers recently and who "
    "is rested, and the players' records. If the JSON includes a 'model' entry, it is this app's "
    "win-probability model; use it as you see fit. Reply with the id of your choice, your top three ids "
    "(best first, your choice first; fewer if fewer players are available) and one short sentence of reasoning."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer"},
        "top3": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["choice", "top3", "reason"],
    "additionalProperties": False,
}


def params(payload: dict) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
    }


def load(inputs: Path, which: str) -> list[dict]:
    return [json.loads(l) for l in open(inputs / f"items_{which}.jsonl", encoding="utf-8")]


def record(msg, custom_id: str) -> dict:
    text = next((b.text for b in msg.content if b.type == "text"), None)
    u = msg.usage
    return {"custom_id": custom_id, "model": msg.model, "stop_reason": msg.stop_reason, "text": text,
            "usage": {"input": u.input_tokens, "output": u.output_tokens,
                      "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                      "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}}


def append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pilot(inputs: Path, work: Path) -> None:
    """Live calls on the pilot items; resumes (ids already answered are skipped)."""
    import anthropic
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    done = set()
    for arm in ("B", "C"):
        f = work / f"raw_pilot_{arm}.jsonl"
        if f.exists():
            done |= {json.loads(l)["custom_id"] for l in open(f, encoding="utf-8")}
    for it in load(inputs, "pilot"):
        if it.get("excluded"):
            continue
        for arm in ("B", "C"):
            if f"{it['id']}|{arm}" in done:
                continue
            msg = client.messages.create(**params(it[f"payload_{arm}"]))
            rec = record(msg, f"{it['id']}|{arm}")
            append(work / f"raw_pilot_{arm}.jsonl", rec)
            print(arm, it["id"], rec["stop_reason"], rec["usage"], flush=True)


def pilot_cost(work: Path, n_test_calls: int) -> None:
    """Pilot tokens priced at the Batches rate (half of $4 in / $20 out per MTok),
    projected to the test calls (the PREREG cost cap)."""
    recs = [json.loads(l) for arm in ("B", "C") for l in open(work / f"raw_pilot_{arm}.jsonl", encoding="utf-8")]
    per = [(r["usage"]["input"] * 4 + r["usage"]["output"] * 20) / 1e6 / 2 for r in recs if r.get("usage")]
    print(f"pilot calls {len(per)}, mean batch-priced cost per call ${sum(per) / len(per):.4f}, "
          f"projected for {n_test_calls} test calls ${sum(per) / len(per) * n_test_calls:.2f}; "
          f"mean input {sum(r['usage']['input'] for r in recs) / len(recs):.0f}, output {sum(r['usage']['output'] for r in recs) / len(recs):.0f}; "
          f"stop reasons {sorted({r['stop_reason'] for r in recs})}")


def submit(inputs: Path, work: Path, arm: str, retry: bool = False) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    items = [it for it in load(inputs, "test") if not it.get("excluded")]
    tag = arm
    if retry:          # once: the items whose batch request errored
        bad = {json.loads(l)["custom_id"].rsplit("|", 1)[0] for l in open(work / f"raw_test_{arm}.jsonl", encoding="utf-8")
               if json.loads(l).get("error")}
        items = [it for it in items if it["id"] in bad]
        tag = f"{arm}_retry"
        if not items:
            print("nothing to retry")
            return
    reqs = [Request(custom_id=f"{tag}_{i:04d}", params=MessageCreateParamsNonStreaming(**params(it[f"payload_{arm}"])))
            for i, it in enumerate(items)]
    batch = client.messages.batches.create(requests=reqs)
    ids = {f"{tag}_{i:04d}": it["id"] for i, it in enumerate(items)}
    (work / f"batch_{tag}.json").write_text(json.dumps({"batch_id": batch.id, "ids": ids,
                                                         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(arm, batch.id, batch.processing_status, len(reqs), flush=True)


def collect(work: Path, arm: str, retry: bool = False) -> None:
    """Writes raw_test_<arm>.jsonl; a retry batch appends to it (analyze.py keeps
    the last record per item)."""
    import anthropic
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    tag = f"{arm}_retry" if retry else arm
    meta = json.loads((work / f"batch_{tag}.json").read_text())
    b = client.messages.batches.retrieve(meta["batch_id"])
    print(arm, b.processing_status, b.request_counts, flush=True)
    if b.processing_status != "ended":
        return
    out = work / f"raw_test_{arm}.jsonl"
    if out.exists() and not retry:
        print("already collected", out)
        return
    for res in client.messages.batches.results(meta["batch_id"]):
        cid = meta["ids"][res.custom_id] + "|" + arm
        if res.result.type == "succeeded":
            append(out, record(res.result.message, cid))
        else:
            append(out, {"custom_id": cid, "error": res.result.type})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "pilot_cost", "submit", "collect"])
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--n-test-calls", type=int, default=0)
    ap.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--arm", choices=["B", "C"])
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pilot":
        pilot(a.inputs, a.work)
    elif a.cmd == "pilot_cost":
        pilot_cost(a.work, a.n_test_calls)
    elif a.cmd == "submit":
        submit(a.inputs, a.work, a.arm, a.retry)
    else:
        collect(a.work, a.arm, a.retry)


if __name__ == "__main__":
    main()
