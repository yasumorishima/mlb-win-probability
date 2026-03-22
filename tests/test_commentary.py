"""Tests for AI Commentary module (gemini_commentary.py).

Tests prompt building, quality evaluation, caching, and metadata
without requiring a Gemini API key.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from win_probability import full_analysis
from gemini_commentary import (
    PROMPT_REGISTRY,
    PROMPT_VERSION,
    _build_prompt,
    _cache_key,
    _compute_whatif_outcomes,
    _get_current_re24,
    evaluate_commentary_quality,
    generate_commentary,
    get_commentary_metadata,
)


def _sample_result():
    """9th inning drama scenario."""
    return full_analysis(9, "bottom", 2, (1, 1, 1), 0)


# ============================================================
# Prompt Building Tests
# ============================================================

def test_build_prompt_ja_contains_key_data():
    """Japanese prompt should contain WP, LI, RE24, and What-If data."""
    result = _sample_result()
    prompt = _build_prompt(result, lang="JA")
    assert "9回裏" in prompt
    assert "2アウト" in prompt
    assert "同点" in prompt
    assert "勝利確率" in prompt
    assert "レバレッジ指数" in prompt
    assert "期待得点" in prompt
    assert "本塁打" in prompt  # What-If outcome
    assert "三振" in prompt    # What-If outcome


def test_build_prompt_en_contains_key_data():
    """English prompt should contain WP, LI, RE24, and What-If data."""
    result = _sample_result()
    prompt = _build_prompt(result, lang="EN")
    assert "of 9" in prompt
    assert "Outs: 2" in prompt
    assert "Tied" in prompt
    assert "Win Probability" in prompt
    assert "Leverage Index" in prompt
    assert "Run Expectancy" in prompt
    assert "Home Run" in prompt


def test_build_prompt_score_diff_positive():
    """Home leading scenario."""
    result = full_analysis(5, "top", 0, (0, 0, 0), 3)
    prompt_ja = _build_prompt(result, lang="JA")
    assert "3点リード" in prompt_ja
    prompt_en = _build_prompt(result, lang="EN")
    assert "Home leads by 3" in prompt_en


def test_build_prompt_score_diff_negative():
    """Away leading scenario."""
    result = full_analysis(7, "bottom", 1, (1, 1, 0), -2)
    prompt_ja = _build_prompt(result, lang="JA")
    assert "アウェイが2点リード" in prompt_ja


# ============================================================
# What-If & RE24 Tests
# ============================================================

def test_whatif_outcomes_keys():
    """What-If outcomes should include all 6 standard outcomes."""
    outcomes = _compute_whatif_outcomes(5, "top", 1, (1, 0, 0), 0, 4.5)
    expected_keys = {"strikeout", "single", "double", "home_run", "walk", "ground_out"}
    assert set(outcomes.keys()) == expected_keys
    for val in outcomes.values():
        assert "wp" in val
        assert "wpa" in val


def test_whatif_home_run_increases_wp_for_batting_team():
    """A home run should generally increase WP for the batting team."""
    outcomes = _compute_whatif_outcomes(5, "bottom", 0, (1, 0, 0), 0, 4.5)
    assert outcomes["home_run"]["wpa"] > 0


def test_get_current_re24():
    """RE24 for bases loaded, 0 outs should be highest."""
    loaded_0 = _get_current_re24((1, 1, 1), 0, 4.5)
    empty_2 = _get_current_re24((0, 0, 0), 2, 4.5)
    assert loaded_0 > empty_2
    assert loaded_0 > 2.0
    assert empty_2 < 0.2


# ============================================================
# Quality Evaluation Tests
# ============================================================

def test_quality_perfect_score_ja():
    """Commentary mentioning all criteria should score 100."""
    result = _sample_result()
    commentary = (
        "9回裏2アウト満塁という場面、ホームチームの勝利確率は87.1%と非常に高い状況です。"
        "レバレッジ指数は11.19と超高プレッシャーの場面で、この一打で試合が決まります。"
        "期待得点は1.541点と高く、得点の可能性は十分です。"
        "ここでシングルヒットが出れば勝利確率は一気に跳ね上がります。"
        "エンドランも検討に値する作戦です。"
    )
    quality = evaluate_commentary_quality(commentary, result, lang="JA")
    assert quality["score"] == 100
    assert quality["pass"] is True
    assert all(quality["checks"].values())


def test_quality_empty_commentary_scores_low():
    """Very short commentary should fail length check."""
    result = _sample_result()
    commentary = "いい場面です。"
    quality = evaluate_commentary_quality(commentary, result, lang="JA")
    assert quality["checks"]["appropriate_length"] is False
    assert quality["score"] < 100


def test_quality_no_data_reference():
    """Commentary that doesn't reference any data should score low."""
    result = _sample_result()
    commentary = (
        "さあ、盛り上がってきました！球場全体が歓声に包まれています。"
        "ファンの声援が選手たちの背中を押します。"
        "果たしてどうなるのでしょうか。ドラマチックな展開が待っています。"
    )
    quality = evaluate_commentary_quality(commentary, result, lang="JA")
    assert quality["checks"]["mentions_wp"] is False
    assert quality["checks"]["mentions_li"] is False
    assert quality["score"] < 60
    assert quality["pass"] is False


def test_quality_en_perfect():
    """English commentary mentioning all criteria should pass."""
    result = _sample_result()
    commentary = (
        "We're at a critical juncture. The home team's win probability stands at 87.1%, "
        "and the leverage index of 11.19 tells us this is a game-defining moment. "
        "With a run expectancy of 1.541, the scoring potential is enormous. "
        "A single here would push win probability above 90%, while a strikeout "
        "would still leave the home team in a strong position."
    )
    quality = evaluate_commentary_quality(commentary, result, lang="EN")
    assert quality["pass"] is True
    assert quality["score"] >= 80


def test_quality_checks_tactics_only_when_present():
    """If no tactics recommended, mentions_tactics should auto-pass."""
    result = full_analysis(1, "top", 0, (0, 0, 0), 0)
    commentary = "Game just started. Win probability is roughly even."
    quality = evaluate_commentary_quality(commentary, result, lang="EN")
    # No recommended tactics at game start = auto-pass
    assert quality["checks"]["mentions_tactics"] is True


# ============================================================
# Cache Tests
# ============================================================

def test_cache_key_unique():
    """Different game states should produce different cache keys."""
    r1 = full_analysis(9, "bottom", 2, (1, 1, 1), 0)
    r2 = full_analysis(1, "top", 0, (0, 0, 0), 0)
    assert _cache_key(r1, "JA") != _cache_key(r2, "JA")


def test_cache_key_same_state():
    """Same game state + same lang should produce same cache key."""
    r1 = full_analysis(5, "top", 1, (1, 0, 0), 2)
    r2 = full_analysis(5, "top", 1, (1, 0, 0), 2)
    assert _cache_key(r1, "JA") == _cache_key(r2, "JA")


def test_cache_key_differs_by_lang():
    """Same state but different lang should differ."""
    r = full_analysis(5, "top", 1, (1, 0, 0), 2)
    assert _cache_key(r, "JA") != _cache_key(r, "EN")


# ============================================================
# Generation Tests (mocked API)
# ============================================================

def test_generate_commentary_no_key_returns_none():
    """Without API key, generate_commentary should return None."""
    result = _sample_result()
    commentary = generate_commentary(result, lang="JA", api_key=None, track=False)
    assert commentary is None


def test_generate_commentary_uses_cache():
    """If cache has entry, should return cached value without API call."""
    result = _sample_result()
    ck = _cache_key(result, "JA")
    cache = {ck: "Cached commentary text"}

    commentary = generate_commentary(result, lang="JA", api_key="fake-key", cache=cache, track=False)
    assert commentary == "Cached commentary text"


@patch("gemini_commentary.genai")
def test_generate_commentary_stores_in_cache(mock_genai):
    """Generated commentary should be stored in cache."""
    mock_response = MagicMock()
    mock_response.text = "Generated AI commentary with 勝利確率 and レバレッジ"
    mock_response.usage_metadata = None
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client

    result = _sample_result()
    cache = {}
    commentary = generate_commentary(
        result, lang="JA", api_key="fake-key", cache=cache, track=False,
    )

    assert commentary == "Generated AI commentary with 勝利確率 and レバレッジ"
    ck = _cache_key(result, "JA")
    assert cache[ck] == commentary


@patch("gemini_commentary.genai")
def test_generate_commentary_tracks_token_usage(mock_genai):
    """Token usage from response should be extractable."""
    mock_usage = MagicMock()
    mock_usage.total_token_count = 542
    mock_response = MagicMock()
    mock_response.text = "Commentary text"
    mock_response.usage_metadata = mock_usage
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client

    result = _sample_result()
    commentary = generate_commentary(
        result, lang="JA", api_key="fake-key", track=False,
    )
    assert commentary == "Commentary text"


# ============================================================
# Metadata & Registry Tests
# ============================================================

def test_prompt_version_exists_in_registry():
    """Current PROMPT_VERSION must exist in PROMPT_REGISTRY."""
    assert PROMPT_VERSION in PROMPT_REGISTRY


def test_prompt_registry_has_required_fields():
    """Each registry entry must have description and date."""
    for version, meta in PROMPT_REGISTRY.items():
        assert "description" in meta, f"{version} missing description"
        assert "date" in meta, f"{version} missing date"


def test_get_commentary_metadata():
    """Metadata should contain prompt version, model, and quality criteria."""
    meta = get_commentary_metadata()
    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["model"] == "gemini-2.5-flash"
    assert meta["quality_threshold"] == 60
    assert len(meta["quality_criteria"]) == 6
    assert PROMPT_VERSION in meta["available_versions"]
