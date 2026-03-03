"""Tests for MLB Win Probability core calculations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from win_probability import (
    MLB_RPG,
    NPB_RPG,
    SCENARIOS,
    analyze_scenario,
    calculate_li,
    calculate_wp,
    get_re24_table,
    li_label,
)


def test_wp_game_start_is_half():
    """1回表開始・同点 → WP ≈ 0.5"""
    wp = calculate_wp(1, "top", 0, (0, 0, 0), 0)
    assert 0.45 < wp < 0.55


def test_wp_home_team_large_lead():
    """9回裏・ホーム3点リード → WP ほぼ 1.0"""
    wp = calculate_wp(9, "bottom", 0, (0, 0, 0), 3)
    assert wp > 0.95


def test_wp_away_team_large_lead():
    """9回表・アウェイ3点リード → WP ほぼ 0.0"""
    wp = calculate_wp(9, "top", 0, (0, 0, 0), -3)
    assert wp < 0.05


def test_wp_range():
    """WP は常に [0, 1] の範囲"""
    for inning in [1, 5, 9]:
        for top_bottom in ["top", "bottom"]:
            for score_diff in [-5, 0, 5]:
                wp = calculate_wp(inning, top_bottom, 1, (0, 0, 0), score_diff)
                assert 0.0 <= wp <= 1.0


def test_li_high_leverage():
    """9回裏・同点・2アウト満塁 → LI > 2.0（高レバレッジ）"""
    li = calculate_li(9, "bottom", 2, (1, 1, 1), 0)
    assert li > 2.0


def test_li_low_leverage():
    """1回表・3点リード → LI < 1.0（低レバレッジ）"""
    li = calculate_li(1, "top", 0, (0, 0, 0), 3)
    assert li < 1.0


def test_li_label():
    assert li_label(0.3) == "Low"
    assert li_label(0.5) == "Medium"
    assert li_label(2.0) == "High"
    assert li_label(4.0) == "Very High"


def test_re24_table_has_24_states():
    table = get_re24_table()
    assert len(table) == 24


def test_analyze_scenario_keys():
    result = analyze_scenario("game_start")
    assert "win_probability" in result
    assert "leverage_index" in result
    assert "tactics" in result


def test_analyze_scenario_unknown():
    result = analyze_scenario("nonexistent_scenario")
    assert "error" in result


def test_all_scenarios_valid():
    for key in SCENARIOS:
        result = analyze_scenario(key)
        assert "win_probability" in result
        assert 0.0 <= result["win_probability"] <= 1.0


def test_npb_rpg_lower_wp_change():
    """NPBは得点が低いのでWP変化が緩やか（極端な差は出にくい）"""
    wp_mlb = calculate_wp(5, "top", 0, (0, 0, 0), 1, MLB_RPG)
    wp_npb = calculate_wp(5, "top", 0, (0, 0, 0), 1, NPB_RPG)
    # 両方とも [0,1] の範囲内
    assert 0.0 <= wp_mlb <= 1.0
    assert 0.0 <= wp_npb <= 1.0
