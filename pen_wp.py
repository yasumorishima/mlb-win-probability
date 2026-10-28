"""
Win probability of finished games 2022-2026 from the player layer with the
late-inning bullpen by usage, recalibrated (PEN_RECAL). Pre-registered and
scored on 2026: better than the team layer and the player layer v1
(research/player_layer/RESULTS_PEN_2026.md). Used by the look-back view;
live and the API keep the team layer.

Files: research/player_layer/published/pen_wp_<season>.parquet
(game_pk, at_bat_index, wp = home WP before that plate appearance),
built by research/player_layer/build_pen_wp.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent / "research/player_layer/published"
SEASONS = (2022, 2023, 2024, 2025, 2026)


@lru_cache(maxsize=len(SEASONS))
def _season(season: int) -> dict:
    p = DIR / f"pen_wp_{season}.parquet"
    if season not in SEASONS or not p.exists():
        return {}
    d = pd.read_parquet(p)
    out: dict = {}
    for g, i, w in zip(d.game_pk.to_numpy(), d.at_bat_index.to_numpy(), d.wp.to_numpy()):
        out.setdefault(int(g), {})[int(i)] = float(w)
    return out


def game(game_pk: int, game_date: str | None) -> dict[int, float]:
    """{at_bat_index: home WP before it} for a regular-season game, or {}."""
    try:
        season = int(str(game_date)[:4])
    except (TypeError, ValueError):
        return {}
    return _season(season).get(int(game_pk), {})


def apply_to_series(series: list[dict], at_bats: list[int | None], wp: dict[int, float]) -> tuple[list[dict], int]:
    """series[k] (k < len(at_bats)) is the state before plate appearance k; replace
    its WP with PEN_RECAL where available (the final point, the game's end, stays).
    Returns the new series and the number of points replaced."""
    if not wp:
        return series, 0
    out = [dict(p) for p in series]
    n = 0
    for k, ab in enumerate(at_bats):
        if k < len(out) and ab is not None and ab in wp:
            out[k]["wp"] = wp[ab]
            n += 1
    for k in range(1, len(out)):
        out[k]["delta"] = float(out[k]["wp"] - out[k - 1]["wp"])
    return out, n
