"""
Build correct game-state tables from the MLB Stats API (keyless).

Replaces scripts/fetch_game_states.py, which produced unusable data:
it read `about.outs` (that key does not exist, so every row had outs=0)
and took runners only from `runners[].movement.originBase` (runners who
did not move were dropped, so 77% of rows said "bases empty").

Here the state before each plate appearance is the state after the
previous plate appearance of the same half-inning:
  outs   = previous play's count.outs (0 at the start of a half)
  bases  = previous play's matchup.postOnFirst/Second/Third
  extra innings (regular season 2020+) start with a runner on second.
Every row is checked against invariants and the build stops if a game
breaks them.

Outputs (parquet, one file per season):
  pa_<season>.parquet      one row per plate appearance (pre-PA state)
  run_<season>.parquet     one row per steal / caught-stealing / pickoff /
                           step-off event, with runner, base, count, pitcher,
                           catcher (tracked through substitutions), and the
                           number of disengagements earlier in the same PA
  pitch_<season>.parquet   one row per pitch: count before the pitch and the
                           pitch result code (for the count-level layer)

Usage:
  python scripts/build_states.py --season 2025 --out data/states
  python scripts/build_states.py --season 2026 --out data/states --until 2026-09-28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://statsapi.mlb.com/api"
UA = {"User-Agent": "mlb-win-probability/2.0 (github.com/yasumorishima/mlb-win-probability)"}

STEAL_TYPES = ("stolen_base_", "caught_stealing_", "pickoff_caught_stealing_")


class StateError(Exception):
    """A game whose derived states break an invariant."""


def fetch_json(url: str, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def season_games(season: int, until: str | None = None) -> list[dict]:
    """Final regular-season games of a season (schedule endpoint)."""
    end = until or f"{season}-11-30"
    url = (f"{BASE}/v1/schedule?sportId=1&gameType=R&startDate={season}-03-01"
           f"&endDate={end}")
    games = []
    for d in fetch_json(url).get("dates", []):
        for g in d["games"]:
            if g.get("status", {}).get("codedGameState") != "F":
                continue  # final only (drops postponed / suspended-not-finished)
            games.append({"game_pk": g["gamePk"], "date": d["date"]})
    # a suspended game appears on two dates with the same gamePk
    seen, out = set(), []
    for g in games:
        if g["game_pk"] not in seen:
            seen.add(g["game_pk"])
            out.append(g)
    return out


def _bases_after(play: dict) -> tuple[int | None, int | None, int | None]:
    m = play.get("matchup", {})
    return tuple((m.get(k) or {}).get("id") for k in
                 ("postOnFirst", "postOnSecond", "postOnThird"))


def _starting_catchers(feed: dict) -> dict[str, int]:
    """Starting catcher per side ('home'/'away') from the boxscore.

    `battingOrder` is the lineup at the END of the game, so a catcher who
    was replaced is not in it. Starters are the players whose battingOrder
    code ends in "00" (100, 200, ... 900); substitutes are 101, 801, ...
    """
    out = {}
    for side in ("home", "away"):
        team = feed["liveData"]["boxscore"]["teams"][side]
        for p in team["players"].values():
            bo = str(p.get("battingOrder", ""))
            if len(bo) == 3 and bo.endswith("00"):
                pos = (p.get("allPositions") or [{}])[0].get("abbreviation")
                if pos == "C":
                    if side in out:
                        raise StateError(f"two starting catchers for {side}")
                    out[side] = p["person"]["id"]
        if side not in out:
            raise StateError(f"no starting catcher for {side}")
    return out


def parse_game(feed: dict, game_pk: int, date: str) -> tuple[list, list, list]:
    gd = feed["gameData"]
    season = int(gd["game"]["season"])
    ls = feed["liveData"]["linescore"]
    sched_inn = int(ls.get("scheduledInnings", 9))
    home_id = gd["teams"]["home"]["id"]
    away_id = gd["teams"]["away"]["id"]
    final_home = ls["teams"]["home"]["runs"]
    final_away = ls["teams"]["away"]["runs"]
    if final_home == final_away:
        raise StateError(f"{game_pk}: final score tied")
    home_won = int(final_home > final_away)

    try:
        catcher = _starting_catchers(feed)
    except StateError as e:
        raise StateError(f"{game_pk}: {e}") from None
    plays = feed["liveData"]["plays"]["allPlays"]
    pa_rows, run_rows, pitch_rows = [], [], []

    prev_half = None
    outs = 0
    bases: tuple = (None, None, None)
    home_score = away_score = 0

    for play in plays:
        if not play.get("about", {}).get("isComplete", False):
            continue
        ab = play["about"]
        inning = ab["inning"]
        half = "top" if ab["isTopInning"] else "bottom"
        fielding = "home" if half == "top" else "away"
        if (inning, half) != prev_half:
            outs = 0
            bases = (None, None, None)
            if season >= 2020 and inning > sched_inn:
                # automatic runner on second; identity from the first movement
                ghost = None
                for r in play.get("runners", []):
                    if r["movement"].get("start") == "2B" and r["movement"].get("originBase") == "2B":
                        ghost = r["details"]["runner"]["id"]
                        break
                bases = (None, ghost if ghost else -1, None)
            prev_half = (inning, half)

        pre = dict(game_pk=game_pk, date=date, season=season,
                   home_team=home_id, away_team=away_id,
                   scheduled_innings=sched_inn, inning=inning, half=half,
                   at_bat_index=ab["atBatIndex"],
                   outs=outs,
                   on_1b=bases[0], on_2b=bases[1], on_3b=bases[2],
                   home_score=home_score, away_score=away_score,
                   batter=play["matchup"]["batter"]["id"],
                   pitcher=play["matchup"]["pitcher"]["id"],
                   home_won=home_won)

        # Walk the events of this PA in order.
        # The feed records a steal / caught stealing as a non-pitch "action"
        # placed right AFTER the pitch it happened on, carrying the count
        # after that pitch. So for a steal on a pitch, the state the runner
        # and the defense faced is the one in force before the preceding
        # pitch: `at_pitch` keeps that snapshot. Pickoffs and step-offs are
        # not on a pitch and use the current state.
        disengage = 0                       # pickoff attempts + step-offs so far
        count = {"balls": 0, "strikes": 0}  # count now
        ev_outs = outs                      # outs now
        at_pitch = None                     # state before the most recent pitch
        for e in play.get("playEvents", []):
            det = e.get("details", {}) or {}
            et = det.get("eventType") or ""
            etype = e.get("type")
            if e.get("isPitch"):
                at_pitch = dict(balls=count["balls"], strikes=count["strikes"],
                                outs=ev_outs, disengagements=disengage,
                                pitch_number=e.get("pitchNumber"))
                pitch_rows.append(dict(
                    game_pk=game_pk, season=season, at_bat_index=ab["atBatIndex"],
                    pitch_number=e.get("pitchNumber"),
                    balls=count["balls"], strikes=count["strikes"],
                    code=(det.get("call") or {}).get("code"),
                    is_in_play=bool(det.get("isInPlay")),
                ))
                count = {"balls": e["count"]["balls"], "strikes": e["count"]["strikes"]}
                continue
            pos = (e.get("position") or {}).get("abbreviation")
            if pos == "C" and e.get("player") and etype == "action":
                catcher[fielding] = e["player"]["id"]

            is_attempt = etype in ("pickoff", "stepoff")
            is_runner_event = et.startswith(STEAL_TYPES) or et.startswith("pickoff_")
            if is_attempt or is_runner_event:
                on_pitch = et.startswith(("stolen_base_", "caught_stealing_"))
                snap = at_pitch if (on_pitch and at_pitch) else dict(
                    balls=count["balls"], strikes=count["strikes"], outs=ev_outs,
                    disengagements=disengage, pitch_number=None)
                if et:
                    kind = et  # pickoff out / error / steal outcome
                elif etype == "stepoff":
                    kind = "stepoff"
                else:
                    kind = f"pickoff_attempt_{det.get('code')}"
                run_rows.append(dict(
                    game_pk=game_pk, date=date, season=season,
                    at_bat_index=ab["atBatIndex"], event_index=e.get("index"),
                    event_type=kind, runner=(e.get("player") or {}).get("id"),
                    inning=inning, half=half, outs_before=snap["outs"],
                    balls=snap["balls"], strikes=snap["strikes"],
                    disengagements_before=snap["disengagements"],
                    on_pitch=bool(on_pitch and at_pitch),
                    pitch_number=snap["pitch_number"],
                    pitcher=play["matchup"]["pitcher"]["id"],
                    batter=play["matchup"]["batter"]["id"],
                    catcher=catcher.get(fielding),
                    # score at the start of the PA (runs that scored earlier in
                    # the same PA on a wild pitch etc. are not reflected)
                    home_score=home_score, away_score=away_score,
                    is_out=bool(det.get("isOut")),
                    description=det.get("description"),
                ))
                if is_attempt:
                    disengage += 1
            if det.get("isOut") and "count" in e:
                ev_outs = min(e["count"].get("outs", ev_outs), 3)
        last_count = count

        res = play.get("result", {})
        outs_after = play["count"]["outs"]
        new_home = res.get("homeScore", home_score)
        new_away = res.get("awayScore", away_score)
        pa_rows.append(dict(pre, event_type=res.get("eventType"),
                            outs_after=outs_after,
                            runs_on_play=(new_home - home_score) + (new_away - away_score),
                            final_balls=last_count["balls"], final_strikes=last_count["strikes"]))
        home_score, away_score = new_home, new_away
        outs = outs_after
        bases = _bases_after(play)

    if (home_score, away_score) != (final_home, final_away):
        raise StateError(f"{game_pk}: replayed score {home_score}-{away_score} "
                         f"!= final {final_home}-{final_away}")
    _check_invariants(pa_rows, game_pk)
    _check_catchers(run_rows, plays, game_pk)
    return pa_rows, run_rows, pitch_rows


def _check_invariants(rows: list[dict], game_pk: int) -> None:
    halves: dict = {}
    for r in rows:
        if r["outs"] not in (0, 1, 2):
            raise StateError(f"{game_pk}: outs before PA = {r['outs']}")
        halves.setdefault((r["inning"], r["half"]), []).append(r)
    for (inn, half), rs in halves.items():
        if rs[0]["outs"] != 0:
            raise StateError(f"{game_pk}: {inn} {half} does not start at 0 outs")
        for a, b in zip(rs, rs[1:]):
            if b["outs"] < a["outs"]:
                raise StateError(f"{game_pk}: outs decreased inside {inn} {half}")
        sched = rs[0]["scheduled_innings"]
        if rs[0]["season"] >= 2020 and inn > sched and rs[0]["on_2b"] is None:
            raise StateError(f"{game_pk}: extra inning {inn} {half} without runner on 2B")


def _check_catchers(run_rows: list[dict], plays: list, game_pk: int) -> None:
    """Caught-stealing credits name the catcher; the tracked one must match."""
    credited = {}
    for p in plays:
        for r in p.get("runners", []):
            if r["details"].get("eventType", "").startswith("caught_stealing_"):
                for c in r.get("credits", []):
                    if c["position"]["abbreviation"] == "C":
                        credited[(p["atBatIndex"], r["details"].get("playIndex"))] = c["player"]["id"]
    for row in run_rows:
        if row["event_type"].startswith("caught_stealing_"):
            key = (row["at_bat_index"], row["event_index"])
            if key in credited and credited[key] != row["catcher"]:
                raise StateError(f"{game_pk}: tracked catcher {row['catcher']} != credited {credited[key]}")


def build_season(season: int, out: Path, until: str | None, workers: int,
                 limit: int | None = None) -> None:
    games = season_games(season, until)
    if limit:
        games = games[:limit]
    print(f"{season}: {len(games)} final regular-season games", flush=True)

    def work(g):
        feed = fetch_json(f"{BASE}/v1.1/game/{g['game_pk']}/feed/live")
        try:
            return parse_game(feed, g["game_pk"], g["date"]), None
        except StateError as e:
            return None, str(e)

    pa, run, pitch, bad = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (res, err) in enumerate(ex.map(work, games), 1):
            if err:
                bad.append(err)
            else:
                pa += res[0]; run += res[1]; pitch += res[2]
            if i % 200 == 0:
                print(f"  {i}/{len(games)} games, rejected {len(bad)}", flush=True)

    print(f"{season}: rejected {len(bad)} games", flush=True)
    for b in bad[:20]:
        print("   ", b)
    # fail closed if more than 1% of games break an invariant
    if len(bad) > 0.01 * max(len(games), 1):
        sys.exit(f"{season}: {len(bad)}/{len(games)} games rejected (>1%) - not writing")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pa).to_parquet(out / f"pa_{season}.parquet", index=False)
    pd.DataFrame(run).to_parquet(out / f"run_{season}.parquet", index=False)
    pd.DataFrame(pitch).to_parquet(out / f"pitch_{season}.parquet", index=False)
    print(f"{season}: wrote pa {len(pa)} / run {len(run)} / pitch {len(pitch)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/states"))
    ap.add_argument("--until", default=None, help="YYYY-MM-DD last date to include")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N games (smoke test)")
    a = ap.parse_args()
    build_season(a.season, a.out, a.until, a.workers, a.limit)


if __name__ == "__main__":
    main()
