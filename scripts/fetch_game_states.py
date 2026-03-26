"""
Fetch MLB play-by-play game states for WP model validation.

Uses MLB Stats API (free, no auth required).
Outputs CSV with one row per at-bat, including pre-play game state and outcome.
Supports resume: skips already-fetched games if output file exists.
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path


def _log_elapsed(label: str, start: float, budget_min: int = 240):
    elapsed_min = (time.time() - start) / 60
    print(f"  [{label}] elapsed: {elapsed_min:.1f} min / {budget_min} min budget")
    if elapsed_min > budget_min * 0.8:
        print(f"  WARNING: {label} used {elapsed_min:.0f}/{budget_min} min "
              f"({elapsed_min / budget_min * 100:.0f}%) -- timeout risk!")

BASE_URL = "https://statsapi.mlb.com/api"
DATA_DIR = Path(__file__).parent.parent / "data"

FIELDNAMES = [
    "game_pk", "date", "home_team", "away_team", "home_won",
    "play_idx", "inning", "half_inning", "outs",
    "runner_1b", "runner_2b", "runner_3b",
    "home_score", "away_score", "score_diff",
    "home_score_after", "away_score_after", "event",
]


def fetch_json(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARN: failed {url}: {e}", file=sys.stderr)
                return None


def get_schedule(start_date: str, end_date: str) -> list[dict]:
    """Get all Final regular-season games in the date range."""
    url = (f"{BASE_URL}/v1/schedule?startDate={start_date}"
           f"&endDate={end_date}&sportId=1&gameType=R")
    data = fetch_json(url)
    if not data:
        return []

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            status = g["status"]["detailedState"]
            if status != "Final":
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            games.append({
                "gamePk": g["gamePk"],
                "date": date_entry["date"],
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
            })
    return games


def extract_play_states(game_pk: int, game_info: dict) -> list[dict]:
    """Extract pre-play game states from a single game's play-by-play."""
    data = fetch_json(f"{BASE_URL}/v1.1/game/{game_pk}/feed/live")
    if not data:
        return []

    all_plays = (data.get("liveData", {})
                 .get("plays", {})
                 .get("allPlays", []))

    home_won = 1 if game_info["home_score"] > game_info["away_score"] else 0

    states = []
    prev_home = 0
    prev_away = 0

    for i, play in enumerate(all_plays):
        if not play.get("about", {}).get("isComplete", False):
            continue

        about = play["about"]
        result = play.get("result", {})
        runners_list = play.get("runners", [])

        inning = about.get("inning", 1)
        is_top = about.get("isTopInning", True)
        half_inning = "top" if is_top else "bottom"
        outs = min(about.get("outs", 0), 2)

        # Runners at START of play (originBase = where they stood before)
        r1, r2, r3 = 0, 0, 0
        for r in runners_list:
            origin = r.get("movement", {}).get("originBase")
            if origin == "1B":
                r1 = 1
            elif origin == "2B":
                r2 = 1
            elif origin == "3B":
                r3 = 1

        score_diff = prev_home - prev_away
        home_after = result.get("homeScore", prev_home)
        away_after = result.get("awayScore", prev_away)

        states.append({
            "game_pk": game_pk,
            "date": game_info["date"],
            "home_team": game_info["home_team"],
            "away_team": game_info["away_team"],
            "home_won": home_won,
            "play_idx": i,
            "inning": inning,
            "half_inning": half_inning,
            "outs": outs,
            "runner_1b": r1,
            "runner_2b": r2,
            "runner_3b": r3,
            "home_score": prev_home,
            "away_score": prev_away,
            "score_diff": score_diff,
            "home_score_after": home_after,
            "away_score_after": away_after,
            "event": result.get("event", ""),
        })

        prev_home = home_after
        prev_away = away_after

    return states


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch MLB play-by-play data for WP validation")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--batch-days", type=int, default=7,
                        help="Days per schedule API batch")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between game feed fetches")
    args = parser.parse_args()

    t0 = time.time()

    DATA_DIR.mkdir(exist_ok=True)
    year = args.start_date[:4]
    output = args.output or str(DATA_DIR / f"play_states_{year}.csv")

    # --- Fetch schedule in weekly batches ---
    all_games = []
    current = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    print(f"Fetching schedule: {args.start_date} to {args.end_date}")
    while current <= end:
        batch_end = min(current + timedelta(days=args.batch_days - 1), end)
        s = current.strftime("%Y-%m-%d")
        e = batch_end.strftime("%Y-%m-%d")
        games = get_schedule(s, e)
        all_games.extend(games)
        print(f"  {s} to {e}: {len(games)} Final games")
        current = batch_end + timedelta(days=1)
        time.sleep(0.3)

    print(f"Total games to process: {len(all_games)}")
    _log_elapsed("fetch_schedule", t0)

    # --- Resume support: skip already-fetched games ---
    existing_pks = set()
    if Path(output).exists():
        with open(output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_pks.add(int(row["game_pk"]))
        print(f"Resuming: {len(existing_pks)} games already in CSV")

    remaining = [g for g in all_games if g["gamePk"] not in existing_pks]
    print(f"Games to fetch: {remaining.__len__()}")

    # --- Fetch play-by-play for each game ---
    mode = "a" if existing_pks else "w"
    total_plays = 0
    failed = 0

    with open(output, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not existing_pks:
            writer.writeheader()

        for idx, game in enumerate(remaining, 1):
            pk = game["gamePk"]
            states = extract_play_states(pk, game)

            if not states:
                failed += 1
                print(f"  [{idx}/{len(remaining)}] FAIL: gamePk={pk}")
                continue

            for state in states:
                writer.writerow(state)
            total_plays += len(states)

            if idx % 100 == 0 or idx == len(remaining):
                f.flush()
                print(f"  [{idx}/{len(remaining)}] {total_plays} plays "
                      f"({failed} failed)")

            time.sleep(args.delay)

    total_games = len(existing_pks) + len(remaining) - failed
    _log_elapsed("fetch_play_by_play", t0)
    print(f"\nDone: {total_games} games, ~{total_plays} new plays -> {output}")
    if failed:
        print(f"  ({failed} games failed to fetch)")


if __name__ == "__main__":
    main()
