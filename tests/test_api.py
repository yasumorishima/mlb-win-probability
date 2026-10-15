"""FastAPI endpoints (no network, no API key)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import analysis  # noqa: E402
import api  # noqa: E402

client = TestClient(api.app)


def test_root_lists_advice_not_commentary():
    r = client.get("/")
    assert r.status_code == 200
    eps = r.json()["endpoints"]
    assert "/wp/advice" in eps and "/wp/commentary" not in eps


def test_wp_matches_adapter():
    q = dict(inning=7, top_bottom="bottom", outs=1, runner1=1, runner3=1, score_diff=-1)
    r = client.get("/wp", params=q)
    assert r.status_code == 200
    body = r.json()
    ref = analysis.full_analysis(7, "bottom", 1, (1, 0, 1), -1)
    assert body["win_probability"] == ref["win_probability"]
    assert body["leverage_index"] == ref["leverage_index"]
    assert [t["tactic"] for t in body["tactics"]] == [t["tactic"] for t in ref["tactics"]]
    assert len(body["what_if"]) == len(ref["what_if"])


def test_wp_with_count_and_validation():
    q = dict(inning=3, top_bottom="top", outs=0, balls=3, strikes=0)
    assert client.get("/wp", params=q).json()["game_state"]["balls"] == 3
    assert client.get("/wp", params=dict(inning=3, top_bottom="top", outs=0, balls=3)).status_code == 422
    assert client.get("/wp", params=dict(inning=3, top_bottom="top", outs=3)).status_code == 422


def test_advice_is_503_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.get("/wp/advice", params=dict(inning=8, top_bottom="bottom", outs=0, runner1=1))
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_advice_with_key_uses_strategist(monkeypatch):
    pytest.importorskip("pydantic")
    import strategist
    from strategist import Advice, Citation

    seen = {}

    def fake_advise(situation, engine, players):
        seen["engine"] = engine
        adv = Advice(recommendation="steal_2b", side="offense", confidence=0.6, follows_engine=True,
                     reason="Fast runner.", departure_reason="",
                     cited=[Citation(name="breakeven_success_rate",
                                     value=engine["steal_2b"]["breakeven_success_rate"])])
        return adv, strategist.check_citations(adv, {"situation": situation, "engine": engine,
                                                     "players": players})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(strategist, "advise", fake_advise)
    r = client.get("/wp/advice", params=dict(inning=8, top_bottom="bottom", outs=0, runner1=1))
    assert r.status_code == 200
    body = r.json()
    assert body["advice"]["recommendation"] == "steal_2b"
    assert body["citation_problems"] == []
    assert "steal_2b" in seen["engine"]


def test_wpa_endpoint_end_of_half():
    q = dict(before_inning=9, before_top_bottom="top", before_outs=2, before_score_diff=1,
             after_inning=9, after_top_bottom="top", after_outs=3, after_score_diff=1)
    r = client.get("/wp/play", params=q)
    assert r.status_code == 200 and r.json()["wp_after"] == 1.0


def test_re24_and_scenario():
    r = client.get("/re24", params={"half": "bottom"})
    assert r.status_code == 200 and r.json()["count"] == 24
    r = client.get("/wp/scenario", params={"name": "walkoff_chance"})
    assert r.status_code == 200 and r.json()["scenario"] == "Walk-off Chance"
    assert client.get("/wp/scenario", params={"name": "nope"}).status_code == 422


def test_live_endpoint_uses_feed(monkeypatch):
    import live_feed

    state = {"inning": 6, "top_bottom": "top", "outs": 1, "runners": (1, 0, 0), "score_home": 2,
             "score_away": 3, "score_diff": -1, "scheduled_innings": 9, "extra_runner": True,
             "batter_name": "", "pitcher_name": "", "away_team": "A", "home_team": "H",
             "status": "In Progress", "balls": 1, "strikes": 1, "pitch_data": None, "hit_data": None}
    monkeypatch.setattr(live_feed, "get_live_state", lambda pk: dict(state))
    r = client.get("/wp/live/1")
    assert r.status_code == 200
    body = r.json()
    assert body["win_probability"] == analysis.full_analysis(6, "top", 1, (1, 0, 0), -1,
                                                             balls=1, strikes=1)["win_probability"]
    assert body["leverage_label"] in ("Low", "Medium", "High", "Very High")
    monkeypatch.setattr(live_feed, "get_live_state", lambda pk: None)
    assert client.get("/wp/live/1").status_code == 404
