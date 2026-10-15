"""
Claude as a bench strategist.

Replaces gemini_commentary.py. Two uses:

* advise(): given a game situation, the engine's numbers for each tactic,
  and what is known about the players involved, Claude recommends what to
  do on the next pitch and says which numbers it relied on. Every number
  it cites is checked against the inputs (`check_citations`).
* steal_probability(): Claude's estimate that a stolen-base attempt
  succeeds, from the same features a statistical model sees. This is the
  part that can be scored against what actually happened.

The API key is read from ANTHROPIC_API_KEY. Without it the functions
return None; the public demo shows stored results instead of calling the
API, so nobody can spend the key through the app.
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

MODEL = "claude-opus-5-5"
EFFORT = "medium"
PROMPT_VERSION = "s1"

TacticName = Literal["none", "steal_2b", "steal_3b", "sacrifice_bunt", "intentional_walk",
                     "pitching_change", "pinch_hitter", "other"]


class Citation(BaseModel):
    name: str = Field(description="key of the input number, exactly as given")
    value: float = Field(description="the value as given in the input")


class Advice(BaseModel):
    recommendation: TacticName
    side: Literal["offense", "defense"]
    confidence: float = Field(description="0-1, how sure the recommendation is right")
    follows_engine: bool = Field(description="false if this departs from the engine's best option")
    reason: str = Field(description="2-4 sentences, plain language")
    departure_reason: str = Field(description="if follows_engine is false: what the engine does not see; else empty")
    cited: list[Citation]


class StealEstimate(BaseModel):
    p_success: float = Field(description="probability this attempt succeeds, 0-1")
    reason: str = Field(description="one or two sentences")


SYSTEM_ADVICE = """You are the bench coach's analyst for an MLB game. You get the game \
situation, the win-probability engine's numbers for each available tactic, and \
facts about the players involved. Recommend what the team should do before the \
next pitch.

Rules:
- Use only the numbers and facts in the input. Do not add player history, \
injuries, streaks or anything else you were not given. If you would want a fact \
you do not have, say so in the reason instead of assuming it.
- Every number you rely on goes in `cited` with its input key and value exactly as given.
- For a steal, compare the runner's likely success rate with the engine's break-even \
rate; for other tactics compare the engine's win-probability change.
- The engine assumes league-average players. Depart from its best option only when \
the player facts give a concrete reason, and state that reason in `departure_reason`.
- "none" (play it straight) is a valid and frequent answer.
"""

SYSTEM_STEAL = """You estimate whether a stolen-base attempt in an MLB game succeeds. \
You get the situation and facts about the runner, pitcher and catcher, all measured \
before this game. Use only those facts. Give the probability that this attempt is \
successful (runner safe, no caught stealing or pickoff on the play)."""


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


def _schema(model: type[BaseModel]) -> dict:
    """JSON schema for structured outputs (all fields required, no extras)."""
    s = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(s)
    return s


def request_params(kind: str, payload: dict) -> dict:
    """Messages API parameters (shared by live calls and the Batches API)."""
    system, model = (SYSTEM_ADVICE, Advice) if kind == "advice" else (SYSTEM_STEAL, StealEstimate)
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": EFFORT,
                          "format": {"type": "json_schema", "schema": _schema(model)}},
        "messages": [{"role": "user",
                      "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
    }


def parse_response(kind: str, message) -> BaseModel | None:
    """Validate a Messages API response; None on refusal / bad JSON."""
    if getattr(message, "stop_reason", None) in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return None
    model = Advice if kind == "advice" else StealEstimate
    try:
        return model.model_validate_json(text)
    except ValidationError:
        return None


def _call(kind: str, payload: dict):
    client = _client()
    if client is None:
        return None, None
    params = request_params(kind, payload)
    msg = client.beta.messages.create(
        betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
    return parse_response(kind, msg), msg


def advise(situation: dict, engine: dict, players: dict) -> tuple[Advice | None, list[str]]:
    """Recommendation plus the list of citation problems (empty = all cited numbers match)."""
    payload = {"situation": situation, "engine": engine, "players": players}
    adv, _ = _call("advice", payload)
    if adv is None:
        return None, ["no valid response"]
    return adv, check_citations(adv, payload)


def steal_probability(features: dict) -> StealEstimate | None:
    est, _ = _call("steal", features)
    return est


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def check_citations(adv: Advice, payload: dict, tol: float = 5e-3) -> list[str]:
    """Every cited number must exist in the input with the same value
    (rounding to 2-3 decimals and percent form are accepted)."""
    flat = _flatten(payload)
    leaf = {}
    for k, v in flat.items():
        leaf.setdefault(k.split(".")[-1], []).append(v)
    problems = []
    for c in adv.cited:
        cands = flat.get(c.name)
        cands = [cands] if cands is not None else leaf.get(c.name.split(".")[-1], [])
        if not cands:
            problems.append(f"unknown key {c.name}")
            continue
        if not any(abs(c.value - v) <= tol * max(1.0, abs(v)) or abs(c.value - 100 * v) <= 0.5
                   for v in cands):
            problems.append(f"{c.name}={c.value} not in input {cands}")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", adv.reason):
        pct = float(m.group(1))
        if not any(abs(pct - 100 * v) <= 0.6 or abs(pct - v) <= 0.6 for v in flat.values()):
            problems.append(f"reason mentions {pct}% not derivable from input")
    return problems
