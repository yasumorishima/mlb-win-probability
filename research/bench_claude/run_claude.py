"""
Claude's bench calls (PREREG.md). Arms: B = without the model's numbers,
C = with them (payload_B / payload_C of build_items.py). Same system prompt.

  pilot  : python run_claude.py pilot  --inputs I --work W            # live calls on the pilot items (both arms)
  submit : python run_claude.py submit --inputs I --work W --arm B|C  # Batches API, test items not excluded
  collect: python run_claude.py collect --work W --batch <id> --arm B|C

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
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
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
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=300.0)
    for it in load(inputs, "pilot"):
        if it.get("excluded"):
            continue
        for arm in ("B", "C"):
            msg = client.messages.create(**params(it[f"payload_{arm}"]))
            rec = record(msg, f"{it['id']}|{arm}")
            append(work / f"raw_pilot_{arm}.jsonl", rec)
            print(arm, it["id"], rec["stop_reason"], rec["usage"], flush=True)


def submit(inputs: Path, work: Path, arm: str) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    items = [it for it in load(inputs, "test") if not it.get("excluded")]
    reqs = [Request(custom_id=f"{arm}_{i:04d}", params=MessageCreateParamsNonStreaming(**params(it[f"payload_{arm}"])))
            for i, it in enumerate(items)]
    batch = client.messages.batches.create(requests=reqs)
    ids = {f"{arm}_{i:04d}": it["id"] for i, it in enumerate(items)}
    (work / f"batch_{arm}.json").write_text(json.dumps({"batch_id": batch.id, "ids": ids,
                                                         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(arm, batch.id, batch.processing_status, len(reqs), flush=True)


def collect(work: Path, arm: str) -> None:
    import anthropic
    client = anthropic.Anthropic(max_retries=2, timeout=300.0)
    meta = json.loads((work / f"batch_{arm}.json").read_text())
    b = client.messages.batches.retrieve(meta["batch_id"])
    print(arm, b.processing_status, b.request_counts, flush=True)
    if b.processing_status != "ended":
        return
    out = work / f"raw_test_{arm}.jsonl"
    if out.exists():
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
    ap.add_argument("cmd", choices=["pilot", "submit", "collect"])
    ap.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--arm", choices=["B", "C"])
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pilot":
        pilot(a.inputs, a.work)
    elif a.cmd == "submit":
        submit(a.inputs, a.work, a.arm)
    else:
        collect(a.work, a.arm)


if __name__ == "__main__":
    main()
