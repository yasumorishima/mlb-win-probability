"""live_feed.py state reconstruction on a small synthetic /feed/live response.

The state before a play must come from the previous play (count.outs and
matchup.postOn*), as in scripts/build_states.py; extra innings of a 2020+
regular-season game start with a runner on second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_feed import parse_game_plays, parse_live_state  # noqa: E402


def _play(inning, top, outs_after, post, home, away, event="Single", runners=()):
    m = {"batter": {"fullName": "B"}, "pitcher": {"fullName": "P"}}
    for key, on in zip(("postOnFirst", "postOnSecond", "postOnThird"), post):
        if on:
            m[key] = {"id": 1}
    return {"about": {"inning": inning, "isTopInning": top, "isComplete": True},
            "count": {"balls": 0, "strikes": 0, "outs": outs_after},
            "matchup": m, "result": {"event": event, "description": event,
                                     "homeScore": home, "awayScore": away},
            "runners": list(runners), "playEvents": []}


def _feed(plays, game_type="R", season="2024", current=None):
    return {"gameData": {"game": {"type": game_type, "season": season},
                         "teams": {"home": {"name": "H"}, "away": {"name": "A"}},
                         "status": {"detailedState": "In Progress"}},
            "liveData": {"linescore": {"scheduledInnings": 9, "teams": {"home": {"runs": 0},
                                                                        "away": {"runs": 0}}},
                         "plays": {"allPlays": plays, "currentPlay": current or {}}}}


def test_state_before_each_play_comes_from_previous_play():
    plays = [
        _play(1, True, 0, (1, 0, 0), 0, 0),                    # single
        # the plays list no runners: the state must come from the previous
        # play's postOn* (runners[].movement.originBase would see nobody)
        _play(1, True, 0, (1, 1, 0), 0, 0, event="Walk"),
        _play(1, True, 1, (0, 1, 1), 0, 0, event="Field Out"),  # runners advance on the out
        _play(1, True, 1, (1, 1, 1), 0, 1, event="Single"),
        _play(1, True, 3, (0, 0, 0), 0, 1, event="Double Play"),
        _play(1, False, 1, (0, 0, 0), 0, 1, event="Strikeout"),
    ]
    out = parse_game_plays(_feed(plays))
    assert [(p["outs"], p["runners"]) for p in out] == [
        (0, (0, 0, 0)), (0, (1, 0, 0)), (0, (1, 1, 0)), (1, (0, 1, 1)), (1, (1, 1, 1)),
        (0, (0, 0, 0)),
    ]
    assert out[3]["runs_on_play"] == 1 and out[4]["score_diff"] == -1
    assert out[4]["outs_after"] == 3


def test_extra_inning_runner_on_second():
    plays = [_play(10, True, 1, (0, 1, 0), 0, 0, event="Strikeout"),
             _play(10, False, 0, (1, 1, 0), 0, 0, event="Walk")]
    out = parse_game_plays(_feed(plays))
    assert out[0]["runners"] == (0, 1, 0) and out[1]["runners"] == (0, 1, 0)
    assert out[0]["extra_runner"] is True
    post = parse_game_plays(_feed(plays, game_type="D"))      # postseason: no automatic runner
    assert post[0]["runners"] == (0, 0, 0) and post[0]["extra_runner"] is False
    old = parse_game_plays(_feed(plays, season="2019"))
    assert old[0]["runners"] == (0, 0, 0)


def test_live_state_after_third_out_is_next_half():
    last = _play(3, False, 3, (1, 0, 0), 2, 1, event="Groundout")
    s = parse_live_state(_feed([last], current=last))
    assert (s["inning"], s["top_bottom"], s["outs"], s["runners"]) == (4, "top", 0, (0, 0, 0))
    last10 = _play(10, True, 3, (0, 0, 1), 0, 0, event="Flyout")
    s = parse_live_state(_feed([last10], current=last10))
    assert (s["inning"], s["top_bottom"], s["outs"], s["runners"]) == (10, "bottom", 0, (0, 1, 0))


def test_live_state_mid_plate_appearance():
    cur = {"about": {"inning": 5, "isTopInning": False, "isComplete": False},
           "count": {"balls": 2, "strikes": 1, "outs": 1}, "playEvents": []}
    feed = _feed([], current=cur)
    feed["liveData"]["linescore"]["offense"] = {"first": {"id": 1}, "third": {"id": 2}}
    s = parse_live_state(feed)
    assert (s["inning"], s["top_bottom"], s["outs"], s["runners"]) == (5, "bottom", 1, (1, 0, 1))
    assert (s["balls"], s["strikes"]) == (2, 1)


def test_live_state_after_a_play_that_did_not_end_the_half():
    done = _play(2, True, 1, (1, 0, 1), 0, 0, event="Single")
    s = parse_live_state(_feed([done], current=done))
    assert (s["inning"], s["top_bottom"], s["outs"], s["runners"]) == (2, "top", 1, (1, 0, 1))
    assert (s["balls"], s["strikes"]) == (0, 0)


def test_live_wp_statuses():
    from live_feed import live_wp

    base = {"inning": 3, "top_bottom": "top", "outs": 0, "runners": (0, 0, 0), "score_home": 2,
            "score_away": 2, "score_diff": 0, "scheduled_innings": 9, "extra_runner": True,
            "balls": None, "strikes": None}
    for status in ("Postponed", "Cancelled", "Suspended: Rain", "Scheduled"):
        assert live_wp({**base, "status": status})["win_probability"] is None
    assert live_wp({**base, "status": "Final: Tied"})["win_probability"] is None
    assert live_wp({**base, "status": "Final", "score_home": 3})["win_probability"] == 1.0
    assert live_wp({**base, "status": "Final", "score_away": 3})["win_probability"] == 0.0
    assert 0 < live_wp({**base, "status": "In Progress"})["win_probability"] < 1


def _final_feed(status):
    # last play: the third out of the top of the 9th, home team ahead
    last = _play(9, True, 3, (0, 0, 0), 5, 2, event="Groundout")
    f = _feed([last], current=last)
    f["gameData"]["status"]["detailedState"] = status
    f["liveData"]["linescore"]["teams"] = {"home": {"runs": 5}, "away": {"runs": 2}}
    return f


def test_finished_game_has_no_next_half():
    s = parse_live_state(_final_feed("Final"))
    assert (s["inning"], s["top_bottom"]) == (9, "top")
    s = parse_live_state(_final_feed("In Progress"))
    assert (s["inning"], s["top_bottom"], s["outs"]) == (9, "bottom", 0)


def test_has_live_state():
    from live_feed import has_live_state
    for st in ("Final", "Game Over", "Completed Early", "Postponed", "Cancelled", "Suspended: Rain"):
        assert not has_live_state(st), st
    for st in ("In Progress", "Scheduled", "Manager challenge"):
        assert has_live_state(st), st


def test_bad_date_returns_no_games():
    from live_feed import get_todays_games
    assert get_todays_games("2026-10-15 x") == []
