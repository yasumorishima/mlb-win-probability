"""PEN_RECAL win probability for the look-back view (pen_wp.py)."""
import pandas as pd

import pen_wp


def _series(n):
    return [{"i": k, "wp": 0.5, "delta": 0.0} for k in range(n + 1)]


def test_series_point_k_is_the_state_before_plate_appearance_k():
    out, n = pen_wp.apply_to_series(_series(3), [10, 11, 12], {10: 0.1, 11: 0.2, 12: 0.3})
    assert n == 3
    assert [p["wp"] for p in out] == [0.1, 0.2, 0.3, 0.5]       # last point (game end) untouched
    assert abs(out[1]["delta"] - 0.1) < 1e-12 and abs(out[3]["delta"] - 0.2) < 1e-12


def test_missing_plate_appearances_keep_the_old_value():
    out, n = pen_wp.apply_to_series(_series(3), [10, None, 12], {10: 0.1})
    assert n == 1 and [p["wp"] for p in out] == [0.1, 0.5, 0.5, 0.5]
    same, n0 = pen_wp.apply_to_series(_series(2), [1, 2], {})
    assert n0 == 0 and same == _series(2)


def test_files_cover_2022_to_2026_and_match_the_scored_predictions():
    for s in pen_wp.SEASONS:
        assert (pen_wp.DIR / f"pen_wp_{s}.parquet").exists()
    scored = pd.read_parquet(pen_wp.DIR / "pent_2026.parquet", columns=["game_pk", "at_bat_index", "p_PENT_RECAL"]).head(500)
    for g, i, p in zip(scored.game_pk, scored.at_bat_index, scored.p_PENT_RECAL):   # PREREG_PENT, H1 supported
        assert abs(pen_wp.game(int(g), "2026-07-01")[int(i)] - p) <= 5.1e-5


def test_other_seasons_and_bad_dates_give_nothing():
    assert pen_wp.game(1, "2021-06-01") == {}
    assert pen_wp.game(1, None) == {}
