"""
Serve streamlit_app.py with a fake live feed whose state changes on every fetch.

Used on 2026-10-15 to prove that the live view refreshes by itself (PR #10):
without any click the app fetched at 12:36:31, 12:37:07, 12:37:37 UTC, and
stopped when auto-refresh was switched off. No network, no API key.

  streamlit run tools/fake_live_app.py --server.port 8599 --server.headless true
  # watch stdout for "FETCH n HH:MM:SS" lines (one per fetch)

Run from the repository root.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_feed  # noqa: E402

_n = live_feed.__dict__.setdefault("_fake_counter", {"k": 0})


def fake_games(date=None):
    return [{"gamePk": 1, "away_team": "Fake Away", "home_team": "Fake Home",
             "status": "In Progress", "start_time_utc": ""}]


def fake_state(pk):
    _n["k"] += 1
    print("FETCH", _n["k"], time.strftime("%H:%M:%S", time.gmtime()), flush=True)
    return {"inning": 5, "top_bottom": "top", "outs": 1, "runners": (1, 0, 0),
            "score_home": 2, "score_away": _n["k"], "score_diff": 2 - _n["k"],
            "status": "In Progress", "away_team": "Fake Away", "home_team": "Fake Home",
            "batter_name": "fetch #" + str(_n["k"]), "pitcher_name": "p", "balls": 1, "strikes": 1,
            "scheduled_innings": 9, "extra_runner": True}


live_feed.get_todays_games = fake_games
live_feed.get_live_state = fake_state
app = ROOT / "streamlit_app.py"
exec(compile(app.read_text(encoding="utf-8"), str(app), "exec"))
