"""Tests for analysis.py: what the app and the API show, checked against the engine."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import analysis as a  # noqa: E402
from wp_engine import END, STATES  # noqa: E402

ENG = a.get_engine()


def test_wp_matches_engine():
    for args in [(1, "top", 0, (0, 0, 0), 0), (7, "bottom", 1, (1, 1, 0), -1),
                 (9, "bottom", 2, (1, 1, 1), 0), (11, "top", 0, (0, 1, 0), 0)]:
        assert a.home_wp(*args) == ENG.wp(*args)
        r = a.full_analysis(*args)
        assert r["win_probability"] == round(ENG.wp(*args), 4)


def test_wp_game_start_near_half_with_home_edge():
    wp = a.home_wp(1, "top", 0, (0, 0, 0), 0)
    assert 0.5 < wp < 0.56


def test_wp_large_leads_late():
    assert a.home_wp(9, "top", 0, (0, 0, 0), 3) > 0.95
    assert a.home_wp(9, "top", 0, (0, 0, 0), -3) < 0.05


def test_wp_range():
    for inning in [1, 5, 9, 12]:
        for half in a.HALVES:
            for diff in [-5, 0, 5]:
                for outs in range(3):
                    wp = a.home_wp(inning, half, outs, (1, 0, 1), diff)
                    assert 0.0 <= wp <= 1.0


def test_batting_team_wp_is_complement_in_top_half():
    r = a.full_analysis(4, "top", 1, (1, 0, 0), 2)
    assert r["batting_team"] == "away"
    assert r["batting_team_wp"] == pytest.approx(1 - r["win_probability"], abs=1e-4)
    r = a.full_analysis(4, "bottom", 1, (1, 0, 0), 2)
    assert r["batting_team"] == "home"
    assert r["batting_team_wp"] == r["win_probability"]


def test_count_layer():
    base = a.home_wp(6, "bottom", 1, (1, 0, 0), 0)
    assert a.home_wp(6, "bottom", 1, (1, 0, 0), 0, balls=0, strikes=0) == base
    ahead = a.home_wp(6, "bottom", 1, (1, 0, 0), 0, balls=3, strikes=0)
    behind = a.home_wp(6, "bottom", 1, (1, 0, 0), 0, balls=0, strikes=2)
    assert ahead > base > behind
    with pytest.raises(ValueError):
        a.home_wp(6, "bottom", 1, (1, 0, 0), 0, balls=4, strikes=0)


def test_leverage_index_uses_league_swing():
    li = a.leverage_index(9, "bottom", 2, (1, 1, 1), 0)
    raw = ENG.leverage(9, "bottom", 2, (1, 1, 1), 0)
    assert li == pytest.approx(raw / ENG.t["league_swing"])
    assert li > 2.0
    assert a.leverage_index(1, "top", 0, (0, 0, 0), 3) < 1.0


def test_li_label():
    assert a.li_label(0.3) == "Low"
    assert a.li_label(0.5) == "Medium"
    assert a.li_label(2.0) == "High"
    assert a.li_label(4.0) == "Very High"


def test_re24_table_is_engine_run_expectancy():
    for half in a.HALVES:
        table = a.re24_table(half)
        assert len(table) == 24
        for row in table:
            s = (row["outs"], row["runner1"], row["runner2"], row["runner3"])
            dist = ENG.runs[half][s]
            assert row["expected_runs"] == round(float(np.arange(len(dist)) @ dist), 3)
            assert row["p_score"] == round(float(1 - dist[0]), 3)
        by = {(r["runners"], r["outs"]): r["expected_runs"] for r in table}
        for label in a.RUNNER_LABELS.values():
            assert by[(label, 0)] > by[(label, 1)] > by[(label, 2)]
        assert 0.4 < by[("---", 0)] < 0.6
    assert {s for s in STATES} == {(r["outs"], r["runner1"], r["runner2"], r["runner3"])
                                   for r in a.re24_table("top")}


def test_what_if_signs_and_walkoff():
    w = {o["outcome"]: o for o in a.what_if(5, "top", 0, (0, 0, 0), 0)}
    assert w["HR"]["delta"] > w["1B"]["delta"] > 0 > w["K"]["delta"]
    # home WP after a home run with the bases empty, top 5th: away scores one
    assert w["HR"]["home_wp"] == pytest.approx(
        ENG._wp_after(5, "top", (0, 0, 0, 0), 1, 0, 9), abs=1e-4)
    walkoff = {o["outcome"]: o for o in a.what_if(9, "bottom", 1, (0, 1, 1), 0)}
    assert walkoff["HR"]["home_wp"] == 1.0
    assert walkoff["1B"]["home_wp"] > 0.99


def test_tactics_match_engine_and_name_the_side():
    args = (7, "bottom", 0, (1, 0, 0), -1)
    ours = {t["tactic"]: t for t in a.tactics(*args)}
    theirs = {t.tactic: t for t in ENG.tactics(*args)}
    assert set(ours) == set(theirs) == {"steal_2b", "sacrifice_bunt", "intentional_walk"}
    assert ours["steal_2b"]["breakeven"] == round(theirs["steal_2b"].breakeven, 4)
    assert 0 < ours["steal_2b"]["breakeven"] <= 1
    assert ours["steal_2b"]["wp_if"] is None
    assert ours["sacrifice_bunt"]["delta"] == round(theirs["sacrifice_bunt"].delta, 4)
    assert ours["sacrifice_bunt"]["n"] == theirs["sacrifice_bunt"].n
    assert ours["sacrifice_bunt"]["team"] == "home"
    ibb = ours["intentional_walk"]
    assert ibb["side"] == "defense" and ibb["team"] == "away"
    assert ibb["delta"] == round(theirs["intentional_walk"].delta, 4)


def test_bunt_note_flags_count():
    t = {x["tactic"]: x for x in a.tactics(7, "bottom", 0, (1, 0, 0), -1, balls=1, strikes=2)}
    assert "pooled over all counts" in t["sacrifice_bunt"]["note"]
    t = {x["tactic"]: x for x in a.tactics(7, "bottom", 0, (1, 0, 0), -1)}
    assert "pooled over all counts" not in t["sacrifice_bunt"]["note"]


def test_wpa_end_of_half():
    out = a.wpa(a.GameState(9, "top", 2, (0, 0, 0), 0), a.GameState(9, "top", 3, (0, 0, 0), 0))
    assert out["wp_after"] == round(ENG._wp_after(9, "top", END, 0, 0, 9), 4)
    assert out["wpa"] > 0
    # final out of the game with the home team ahead
    end = a.wpa(a.GameState(9, "top", 2, (0, 0, 0), 1), a.GameState(9, "top", 3, (0, 0, 0), 1))
    assert end["wp_after"] == 1.0


def test_wp_after_play():
    # walk-off single
    assert a.wp_after_play(9, "bottom", 1, (1, 0, 0), 1, 0) == 1.0
    # three outs in the top of the 1st, nobody scored: start of the bottom half
    assert a.wp_after_play(1, "top", 3, (0, 0, 0), 0, 0) == pytest.approx(ENG._H(1, "bottom", 0, 9))


def test_game_over_state():
    r = a.full_analysis(9, "bottom", 0, (0, 0, 0), 1)
    assert r["game_over"] and r["win_probability"] == 1.0
    assert r["tactics"] == [] and r["what_if"] == []


def test_input_validation():
    with pytest.raises(ValueError):
        a.full_analysis(1, "middle", 0, (0, 0, 0), 0)
    with pytest.raises(ValueError):
        a.full_analysis(1, "top", 3, (0, 0, 0), 0)
    with pytest.raises(ValueError):
        a.full_analysis(1, "top", 0, (2, 0, 0), 0)
    with pytest.raises(ValueError):
        a.full_analysis(0, "top", 0, (0, 0, 0), 0)


def test_extra_runner_switch():
    with_runner = a.home_wp(10, "top", 0, (0, 1, 0), 0)
    assert a.home_wp(9, "bottom", 2, (0, 0, 0), 0, extra_runner=False) != \
        a.home_wp(9, "bottom", 2, (0, 0, 0), 0)
    assert 0 < with_runner < 1


def test_scenarios():
    for name in a.SCENARIOS:
        r = a.analyze_scenario(name)
        assert {"win_probability", "leverage_index", "tactics", "what_if"} <= set(r)
    with pytest.raises(ValueError):
        a.analyze_scenario("nonexistent_scenario")


def test_advice_inputs_are_citable():
    pytest.importorskip("pydantic")
    from strategist import Advice, Citation, check_citations

    r = a.full_analysis(8, "bottom", 0, (1, 0, 0), 0, balls=1, strikes=0)
    situation, engine, players = a.advice_inputs(r, "JA")
    assert situation["reply_language"] == "Japanese"
    assert engine["steal_2b"]["breakeven_success_rate"] == round(r["tactics"][0]["breakeven"], 3)
    payload = {"situation": situation, "engine": engine, "players": players}
    cited = [Citation(name="engine.steal_2b.breakeven_success_rate",
                      value=engine["steal_2b"]["breakeven_success_rate"]),
             Citation(name="home_wp_now", value=engine["home_wp_now"]),
             Citation(name="leverage_index", value=engine["leverage_index"])]
    adv = Advice(recommendation="none", side="offense", confidence=0.5, follows_engine=True,
                 reason="Play it straight.", departure_reason="", cited=cited)
    assert check_citations(adv, payload) == []


def test_get_advice_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adv, problems, payload = a.get_advice(a.full_analysis(1, "top", 0, (0, 0, 0), 0))
    assert adv is None and problems and "engine" in payload


def test_engine_cache_one_slot_per_flag():
    a.get_engine()
    a.get_engine(True)
    a.get_engine(False)
    before = a._engine.cache_info().misses
    for flag in (True, False, 1, 0, True):
        a.get_engine(flag)
    assert a._engine.cache_info().misses == before
    assert a.get_engine(True) is a.get_engine()


def test_get_advice_catches_api_errors(monkeypatch):
    pytest.importorskip("pydantic")
    import strategist

    def boom(*args):
        raise ConnectionError("network down")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(strategist, "advise", boom)
    adv, problems, _ = a.get_advice(a.full_analysis(5, "top", 0, (1, 0, 0), 0))
    assert adv is None and problems == ["API call failed: ConnectionError"]
