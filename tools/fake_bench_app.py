"""
Serve streamlit_app.py with a recorded game frozen at one moment, to look at
the bench screen without a live game or network (cloud screenshots).

  python tools/fake_bench_app.py <feed.json> <play index>   (prepares the feed)
  FAKE_FEED=<prepared.json> streamlit run tools/fake_bench_app.py --server.headless true

<feed.json> = /api/v1.1/game/<pk>/feed/live of a finished game (not committed).
FAKE_FEED=<feed.json> (unprepared, status Final) serves it for the look-back view.
"""

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(src: str, k: int) -> dict:
    feed = json.load(open(src))
    plays = feed["liveData"]["plays"]["allPlays"]
    f = copy.deepcopy(feed)
    f["liveData"]["plays"]["allPlays"] = plays[:k]
    cp = copy.deepcopy(plays[k])
    prev = plays[k - 1]
    same = prev["about"]["halfInning"] == cp["about"]["halfInning"] and prev["about"]["inning"] == cp["about"]["inning"]
    cp["about"]["isComplete"] = False
    cp["count"] = {"balls": 1, "strikes": 1, "outs": prev["count"]["outs"] if same else 0}
    f["liveData"]["plays"]["currentPlay"] = cp
    f["gameData"]["status"]["detailedState"] = "In Progress"
    top = cp["about"]["isTopInning"]
    bat, fld = ("away", "home") if top else ("home", "away")
    off = {"batter": cp["matchup"]["batter"], "team": {"id": f["gameData"]["teams"][bat]["id"]}}
    if same:
        for b, key in (("first", "postOnFirst"), ("second", "postOnSecond"), ("third", "postOnThird")):
            if key in prev["matchup"]:
                off[b] = prev["matchup"][key]
    f["liveData"]["linescore"]["offense"] = off
    f["liveData"]["linescore"]["teams"] = {"home": {"runs": prev["result"]["homeScore"]},
                                           "away": {"runs": prev["result"]["awayScore"]}}
    f["liveData"]["linescore"]["defense"] = {"batter": {"id": 0}, "team": {"id": f["gameData"]["teams"][fld]["id"]}}
    return f


if __name__ == "__main__" and len(sys.argv) == 3:
    out = Path(sys.argv[1]).with_suffix(f".at{sys.argv[2]}.json")
    out.write_text(json.dumps(prepare(sys.argv[1], int(sys.argv[2]))))
    print(out)
else:
    import bench_live
    import live_feed
    FEED = json.load(open(os.environ["FAKE_FEED"]))
    gd = FEED["gameData"]
    status = gd["status"]["detailedState"]         # a finished game's full feed -> look-back mode
    live_feed.get_todays_games = lambda date=None: [{"gamePk": gd["game"]["pk"], "away_team": gd["teams"]["away"]["name"],
                                                     "home_team": gd["teams"]["home"]["name"], "status": status,
                                                     "start_time_utc": ""}]
    bench_live.get_feed = lambda pk: FEED
    app = ROOT / "streamlit_app.py"
    exec(compile(app.read_text(encoding="utf-8"), str(app), "exec"))
