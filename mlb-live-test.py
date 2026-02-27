#!/usr/bin/env python3
"""
MLB Live Test — calls the local FastAPI (port 8001) to verify live WP calculation.

Meant to run on RPi via cron:
  */5 18-2 * * * /usr/bin/python3 ~/mlb-live-test.py >> ~/mlb-live-test.log 2>&1
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

API = "http://localhost:8001"


def fetch(url: str):
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return {"error": str(e)}


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Get today's games
    schedule = fetch(f"{API}/games/today")
    if "error" in schedule:
        print(f"[{now}] API ERROR: {schedule['error']}")
        return

    games = schedule.get("games", [])
    total = schedule.get("count", 0)
    live_games = [g for g in games if g.get("status") == "In Progress"]

    if not live_games:
        print(f"[{now}] No live games (today total: {total})")
        return

    for g in live_games:
        gpk = g["gamePk"]
        matchup = f"{g['away_team']} @ {g['home_team']}"

        # Call /wp/live/{gamePk} — this computes WP inside Docker
        wp_data = fetch(f"{API}/wp/live/{gpk}")
        if "error" in wp_data:
            print(f"[{now}] LIVE | {matchup} | WP ERROR: {wp_data['error']}")
            continue

        wp = wp_data.get("win_probability")
        if wp is None:
            note = wp_data.get("status_note", "unknown")
            print(f"[{now}] LIVE | {matchup} | {note}")
            continue

        inn = wp_data.get("inning", "?")
        tb = wp_data.get("top_bottom", "?")
        outs = wp_data.get("outs", "?")
        li = wp_data.get("leverage_index", 0)
        li_lbl = wp_data.get("li_label", "")
        score_h = wp_data.get("score_home", 0)
        score_a = wp_data.get("score_away", 0)
        batter = wp_data.get("batter_name", "")
        pitcher = wp_data.get("pitcher_name", "")

        print(
            f"[{now}] LIVE | {matchup} | "
            f"{score_a}-{score_h} | {tb[0].upper()}{inn} {outs}out | "
            f"WP={wp:.1%} LI={li:.2f}({li_lbl}) | "
            f"AB:{batter} P:{pitcher}"
        )


if __name__ == "__main__":
    main()
