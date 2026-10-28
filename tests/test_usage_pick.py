"""bench_replay.usage_pick: the usage-pattern reliever shown in the look-back view."""
import numpy as np

import bench_live as BL
import bench_replay as R


def _setup(monkeypatch):
    u = {11: np.zeros((3, 4)), 22: np.zeros((3, 4))}
    u[11][1, 2] = 30          # 8th inning, ahead by 1-3: pitcher 11
    u[22][1, 1] = 30          # 8th inning, tied: pitcher 22
    monkeypatch.setattr(BL.PU, "usage", lambda team, day, season=None: u)
    opts = {"all": [{"kind": "pitch_change", "option": "pitch_change:11", "wp_offense": 0.30},
                    {"kind": "pitch_change", "option": "pitch_change:22", "wp_offense": 0.40},
                    {"kind": "pitch_change", "option": "pitch_change:33", "wp_offense": 0.20}]}
    return opts


def _tl(home_before, away_before, home_now, away_now):
    base = {"home_id": 1, "away_id": 2, "game_date": "2026-07-01"}
    return [{**base, "inning": 7, "top_bottom": "bottom", "home_score": home_before, "away_score": away_before},
            {**base, "inning": 8, "top_bottom": "top", "home_score": home_now, "away_score": away_now},
            {**base, "inning": 8, "top_bottom": "top", "home_score": home_now, "away_score": away_now}]


def test_pick_uses_the_lead_when_the_half_began(monkeypatch):
    opts = _setup(monkeypatch)
    dec = {"type": "pitch_change", "player": 22}
    # home fields in the top of the 8th, ahead 2-1 when the half began, tied 2-2 at the change
    tl = _tl(2, 1, 2, 2)
    up = R.usage_pick(tl[2], tl, dec, opts)
    assert up["player"] == 11                       # "ahead by 1-3" usage, not "tied"
    assert (up["rank"], up["n"]) == (2, 3)          # model order for the fielding team: 33, 11, 22
    tl = _tl(2, 2, 2, 3)                            # tied when the half began
    assert R.usage_pick(tl[2], tl, dec, opts)["player"] == 22


def test_no_pick_before_the_6th_or_for_other_calls(monkeypatch):
    opts = _setup(monkeypatch)
    tl = _tl(2, 1, 2, 2)
    early = {**tl[2], "inning": 5}
    assert R.usage_pick(early, tl, {"type": "pitch_change", "player": 22}, opts) is None
    assert R.usage_pick(tl[2], tl, {"type": "pinch", "player": 22}, opts) is None
