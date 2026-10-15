"""Tests for strategist.py that never call the API."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import strategist  # noqa: E402
from strategist import Advice, Citation, check_citations, parse_response, request_params  # noqa: E402

PAYLOAD = {
    "situation": {"inning": 8, "half": "bottom", "outs": 0, "score_diff": 0},
    "engine": {"wp_now": 0.688, "steal_2b": {"breakeven": 0.592}},
    "players": {"runner": {"sprint_speed_prev": 29.4}},
}


def _advice(cited, reason="Steal: he is fast."):
    return Advice(recommendation="steal_2b", side="offense", confidence=0.6,
                  follows_engine=True, reason=reason, departure_reason="",
                  cited=[Citation(name=n, value=v) for n, v in cited])


def test_citations_that_match_pass():
    adv = _advice([("engine.steal_2b.breakeven", 0.592), ("sprint_speed_prev", 29.4),
                   ("wp_now", 68.8)])  # percent form accepted
    assert check_citations(adv, PAYLOAD) == []


def test_wrong_value_and_unknown_key_are_flagged():
    adv = _advice([("engine.steal_2b.breakeven", 0.65), ("runner_ops", 0.9)])
    problems = check_citations(adv, PAYLOAD)
    assert any("breakeven" in p for p in problems)
    assert any("unknown key runner_ops" in p for p in problems)


def test_invented_percentage_in_reason_is_flagged():
    adv = _advice([], reason="He succeeds 91% of the time.")
    assert any("91.0%" in p for p in check_citations(adv, PAYLOAD))


def test_request_params_schema_is_closed():
    params = request_params("advice", PAYLOAD)
    schema = params["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert params["model"] == "claude-opus-5-5"
    # the payload is serialized deterministically (cache-friendly)
    assert params["messages"][0]["content"] == json.dumps(PAYLOAD, ensure_ascii=False, sort_keys=True)


def test_parse_response_handles_refusal_and_bad_json():
    refused = SimpleNamespace(stop_reason="refusal", content=[])
    assert parse_response("steal", refused) is None
    bad = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="{}")])
    assert parse_response("steal", bad) is None
    ok = SimpleNamespace(stop_reason="end_turn",
                         content=[SimpleNamespace(type="text", text='{"p_success": 0.8, "reason": "fast"}')])
    assert parse_response("steal", ok).p_success == 0.8


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert strategist.steal_probability({"x": 1}) is None
    adv, problems = strategist.advise({}, {}, {})
    assert adv is None and problems == ["no valid response"]


def test_client_limits_retries_and_timeout(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    import strategist
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    strategist._client()
    assert seen == {"max_retries": 0, "timeout": 60.0}
