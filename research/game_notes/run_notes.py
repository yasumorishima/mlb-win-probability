"""
Generate one short Japanese note per 2026 game with Claude (look-back view).

  env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY="$KEY" \
      python research/game_notes/run_notes.py --inputs IN.jsonl --raw RAW.jsonl [--limit N] [--cap-usd 22] [--workers 8]

- Every raw response is appended to RAW (one JSON line: game_pk, model, usage, stop_reason, text)
  before anything else is done with it; a restart skips games already in RAW (no game is bought twice).
- Cost is summed from `usage` with the model's list price; the run stops before the cap is passed.
- Retries: one retry after 429 / 5xx / connection errors; any other error stops the run (exit 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 10.0 / 1e6          # USD per token (list price)
SYSTEM = """あなたは野球の試合を振り返る解説者です。与えられた JSON（1 試合分のデータ）だけを使って、
日本語で 3〜4 文、全体で 250 字以内の短い解説を 1 段落で書いてください。

決まり:
- 試合の流れと分かれ目（勝率が大きく動いたプレー）を中心に、必要なら終盤の継投に触れる。
- 数字は JSON にある値だけをそのまま使う。スコアは JSON にあるスコア（final、score_before、score_after、score）だけを書く。足し算・引き算などで新しい数字を作らない（「〇ポイント上昇」は書かない。「30% から 83% に」のように JSON の値で書く）。
- 勝率は「（打っているチームの）勝率」のように、どちらのチームの値か分かるように書く。
- 選手名・チーム名は JSON の表記のまま（英語のまま）使う。
- JSON にない事実（けが、記録、順位、采配の意図、選手の調子など）は書かない。推測しない。
- です・ます調。見出しや箇条書きは使わず、本文だけを出力する。"""

_lock = threading.Lock()


def done_ids(raw: Path) -> set[int]:
    if not raw.exists():
        return set()
    return {json.loads(l)["game_pk"] for l in raw.read_text().splitlines() if l.strip()}


def cost_so_far(raw: Path) -> float:
    if not raw.exists():
        return 0.0
    c = 0.0
    for l in raw.read_text().splitlines():
        u = json.loads(l).get("usage") or {}
        c += u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT
    return c


def call(client, obj: dict) -> dict:
    msg = json.dumps(obj, ensure_ascii=False)
    for attempt in (0, 1):
        try:
            r = client.messages.create(model=MODEL, max_tokens=800, system=SYSTEM,
                                       thinking={"type": "between_tools"}, output_config={"effort": "low"},
                                       messages=[{"role": "user", "content": msg}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            return {"game_pk": obj["game_pk"], "model": r.model, "stop_reason": r.stop_reason,
                    "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
                    "text": text}
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            if attempt == 1:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cap-usd", type=float, default=22.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    games = [json.loads(l) for l in a.inputs.read_text().splitlines() if l.strip()]
    have = done_ids(a.raw)
    todo = [g for g in games if g["game_pk"] not in have][: a.limit]
    spent = cost_so_far(a.raw)
    print(f"games {len(games)}, done {len(have)}, todo {len(todo)}, spent ${spent:.4f}", flush=True)
    client = anthropic.Anthropic(max_retries=0, timeout=90)
    state = {"spent": spent, "n": 0, "stop": False}
    per_call_max = 3000 * PRICE_IN + 800 * PRICE_OUT       # a bound on one call, to stop before the cap

    def job(g):
        with _lock:
            if state["stop"] or state["spent"] + per_call_max * a.workers > a.cap_usd:
                state["stop"] = True
                return None
        rec = call(client, g)
        with _lock:
            with a.raw.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            u = rec["usage"]
            state["spent"] += u["input_tokens"] * PRICE_IN + u["output_tokens"] * PRICE_OUT
            state["n"] += 1
            if state["n"] % 100 == 0:
                print(f"done {state['n']}, spent ${state['spent']:.4f}", flush=True)
        return rec

    try:
        with ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(job, g) for g in todo]
            for f in as_completed(futs):
                f.result()
    except Exception as e:  # noqa: BLE001
        print("STOP:", type(e).__name__, str(e)[:300], flush=True)
        sys.exit(1)
    print(f"finished: calls {state['n']}, spent ${state['spent']:.4f}, stopped_by_cap {state['stop']}", flush=True)


if __name__ == "__main__":
    main()
