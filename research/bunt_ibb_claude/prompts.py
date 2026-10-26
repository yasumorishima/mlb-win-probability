"""
Prompts and request parameters of the bunt / walk-candidate study (frozen
with PREREG.md). Self-contained so that the app's strategist.py, and the
frozen SYSTEM_STEAL prompt in it, are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategist import _schema  # noqa: E402  (pure helper, makes no call)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000


class Estimate(BaseModel):
    p_success: float = Field(description="probability of success as defined in the input, 0-1")
    reason: str = Field(description="one or two sentences")


_COMMON = """

You may use anything you know about these players and teams up to the day before the \
given date. Do not use knowledge of this game or of anything after it. If a statistical \
model's estimate is included, it already uses the listed facts; adjust it only for what \
it cannot see.

Return p_success between 0 and 1 and a one- or two-sentence reason."""

SYSTEM = {
    "bunt": ("You estimate the probability that a bunt in an MLB game moves the lead runner up. "
             "The input gives the date, how the plate appearances were selected, what counts as "
             "success, the league success rate this season before that date, the situation before "
             "the pitch, and the batter's and pitcher's records before that date." + _COMMON),
    "obp": ("You estimate the probability that the batter reaches base in a late-inning MLB plate "
            "appearance with first base open and a runner in scoring position. The input gives the "
            "date, how the plate appearances were selected, what counts as success, the league rate "
            "this season before that date, the situation before the plate appearance, and the "
            "batter's and pitcher's records before that date." + _COMMON),
}


def request_params(part: str, payload: dict, effort: str) -> dict:
    """Messages API parameters (live and Batches API). No server-side
    fallback: a fallback would answer with a different model."""
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": SYSTEM[part], "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": _schema(Estimate)}},
        "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
    }


def parse_text(text: str | None, stop_reason: str | None) -> float | None:
    """p_success from a raw response text; None on refusal, truncation,
    invalid JSON or a value outside [0, 1]."""
    if not text or stop_reason != "end_turn":
        return None
    try:
        est = Estimate.model_validate_json(text)
    except ValidationError:
        return None
    return est.p_success if 0.0 <= est.p_success <= 1.0 else None
