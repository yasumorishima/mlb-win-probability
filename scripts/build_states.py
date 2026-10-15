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
import http.client
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
# pitch results that kill the ball (no steal can happen on them):
# foul, foul bunt, foul tip on a bunt, foul pitchout, in play, hit by pitch
DEAD_BALL_CODES = {"F", "L", "O", "R", "X", "D", "E", "H"}
NON_PA_PREFIXES = ("caught_stealing", "pickoff", "stolen_base", "runner_double_play",
                   "other_out", "wild_pitch", "passed_ball", "balk", "defensive_indiff")


class StateError(Exception):
    """A game whose derived states break an invariant."""


def fetch_json(url: str, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                http.client.HTTPException, ConnectionError, OSError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def season_games(season: int, until: str | None = None) -> list[dict]:
    """Final regular-season games of a season (schedule endpoint)."""
    end = f"{season}-11-30"
    if until and until[:4] == str(season):
        end = until          # --until applies only to its own season
    url = (f"{BASE}/v1/schedule?sportId=1&gameType=R&startDate={season}-03-01"
           f"&endDate={end}")
    games = []
    for d in fetch_json(url).get("dates", []):
        for g in d["games"]:
            if str(g.get("season")) != str(season):
                continue  # the schedule endpoint can return other seasons
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
    steal_counts = [0, 0]   # stolen bases, caught stealing (incl. pickoff CS) extracted

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

        # Pass 1 over the PA's events: the state in force BEFORE each event.
        # The count and outs on an event are the values after it, so the
        # state before event k is the one after event k-1.
        events = play.get("playEvents", [])
        pos_of = {e.get("index"): k for k, e in enumerate(events)}
        # runner advances (not the batter, not outs) reset the 2023
        # disengagement counter; collect the event positions where they happen
        advance_at = set()
        for r in play.get("runners", []):
            mv = r.get("movement", {})
            if mv.get("start") and mv.get("end") and not mv.get("isOut") \
                    and mv.get("start") != mv.get("end"):
                k = pos_of.get(r["details"].get("playIndex"))
                if k is not None:
                    advance_at.add(k)
        before = []                              # state before event k
        count = {"balls": 0, "strikes": 0}
        ev_outs = outs
        disengage = 0
        last_pitch = None                        # position of latest pitch so far
        for k, e in enumerate(events):
            det = e.get("details", {}) or {}
            before.append(dict(balls=count["balls"], strikes=count["strikes"],
                               outs=ev_outs, disengagements=disengage,
                               last_pitch=last_pitch, catcher=catcher.get(fielding)))
            if e.get("isPitch"):
                pitch_rows.append(dict(
                    game_pk=game_pk, season=season, at_bat_index=ab["atBatIndex"],
                    pitch_number=e.get("pitchNumber"),
                    balls=count["balls"], strikes=count["strikes"],
                    code=(det.get("call") or {}).get("code"),
                    is_in_play=bool(det.get("isInPlay")),
                ))
                last_pitch = k
            pos = (e.get("position") or {}).get("abbreviation")
            if pos == "C" and e.get("player") and e.get("type") == "action":
                catcher[fielding] = e["player"]["id"]
            # pickoff attempts by the pitcher and step-offs are disengagements;
            # a throw from the catcher is not
            if e.get("type") == "stepoff" or (e.get("type") == "pickoff" and not det.get("fromCatcher")):
                disengage += 1
                run_rows.append(dict(
                    game_pk=game_pk, date=date, season=season,
                    at_bat_index=ab["atBatIndex"], event_index=e.get("index"),
                    row_kind="attempt",
                    event_type="stepoff" if e.get("type") == "stepoff" else f"pickoff_attempt_{det.get('code')}",
                    runner=None, inning=inning, half=half,
                    outs_before=before[k]["outs"], balls=before[k]["balls"],
                    strikes=before[k]["strikes"],
                    disengagements_before=before[k]["disengagements"],
                    on_pitch=False, pitch_number=None,
                    pitcher=play["matchup"]["pitcher"]["id"],
                    pitch_hand=(play["matchup"].get("pitchHand") or {}).get("code"),
                    batter=play["matchup"]["batter"]["id"],
                    catcher=before[k]["catcher"],
                    home_score=home_score, away_score=away_score,
                    is_out=bool(det.get("isOut")),
                    description=det.get("description")))
            if k in advance_at:
                disengage = 0
            if "count" in e:
                count = {"balls": e["count"]["balls"], "strikes": e["count"]["strikes"]}
                ev_outs = min(e["count"].get("outs", ev_outs), 3)
        last_count = {"balls": play["count"]["balls"], "strikes": play["count"]["strikes"]}

        # Pass 2: runner outcomes come from runners[] (authoritative). In the
        # event list an inning-ending caught stealing is an empty "no_pitch"
        # and a steal on the pitch that ends the PA has no event of its own.
        # A steal or caught stealing happens on a pitch: the pitch at
        # playIndex if that event is a pitch, else the latest earlier pitch.
        # Pickoff outcomes happen between pitches: state before the event.
        seen = set()
        n_before = len(run_rows)
        for r in play.get("runners", []):
            rd = r.get("details", {})
            et = rd.get("eventType") or ""
            if not (et.startswith(STEAL_TYPES) or et.startswith("pickoff_")):
                continue
            # eventType is per play: a runner who merely advanced on the same
            # play carries it too. Keep only the runner whose own movement it is.
            if rd.get("movementReason") != f"r_{et}":
                continue
            key = ((rd.get("runner") or {}).get("id"), rd.get("playIndex"), et)
            if key in seen:
                continue
            seen.add(key)
            k = pos_of.get(rd.get("playIndex"))
            if k is None:
                raise StateError(f"{game_pk}: runner event {et} points to a missing playIndex")
            pk_ = None
            if et.startswith(("stolen_base_", "caught_stealing_")):
                cand = k if events[k].get("isPitch") else before[k]["last_pitch"]
                if cand is not None:
                    cdet = events[cand].get("details") or {}
                    going = bool(cdet.get("runnerGoing"))
                    dead = (cdet.get("call") or {}).get("code") in DEAD_BALL_CODES
                    between = [events[j].get("type") for j in range(cand + 1, k)]
                    if going and not dead and \
                            not any(t in ("stepoff", "pickoff", "no_pitch") for t in between):
                        pk_ = cand
            if pk_ is not None:
                snap = before[pk_]
                if not events[k].get("isPitch") and "count" in events[k]:
                    after_pitch = events[pk_]["count"]
                    if (events[k]["count"]["balls"], events[k]["count"]["strikes"]) != \
                            (after_pitch["balls"], after_pitch["strikes"]):
                        raise StateError(f"{game_pk}: steal event count does not follow its pitch "
                                         f"(ab {ab['atBatIndex']}, event {rd.get('playIndex')})")
            else:
                # not on a pitch (pitcher-initiated, pickoff): state before the
                # throw = before the latest pickoff/step-off at or before k when
                # no pitch intervenes, else before the event itself
                j = k
                while j >= 0 and not events[j].get("isPitch"):
                    if events[j].get("type") in ("pickoff", "stepoff"):
                        break
                    j -= 1
                snap = before[j] if (j >= 0 and not events[j].get("isPitch")) else before[k]
            credited = sorted({c["position"]["abbreviation"] for c in r.get("credits", [])})
            run_rows.append(dict(
                game_pk=game_pk, date=date, season=season,
                at_bat_index=ab["atBatIndex"], event_index=rd.get("playIndex"),
                row_kind="outcome", event_type=et, credited_positions=",".join(credited),
                runner=key[0], inning=inning, half=half,
                outs_before=snap["outs"], balls=snap["balls"], strikes=snap["strikes"],
                disengagements_before=snap["disengagements"],
                on_pitch=pk_ is not None,
                pitch_number=events[pk_].get("pitchNumber") if pk_ is not None else None,
                pitcher=play["matchup"]["pitcher"]["id"],
                pitch_hand=(play["matchup"].get("pitchHand") or {}).get("code"),
                batter=play["matchup"]["batter"]["id"],
                catcher=snap["catcher"],
                # score at the start of the PA (runs that scored earlier in
                # the same PA on a wild pitch etc. are not reflected)
                home_score=home_score, away_score=away_score,
                is_out=bool(r.get("movement", {}).get("isOut")),
                description=(events[k].get("details") or {}).get("description")))
        steal_counts[0] += sum(1 for row in run_rows[n_before:] if row["event_type"].startswith("stolen_base_"))
        steal_counts[1] += sum(1 for row in run_rows[n_before:] if row["event_type"].startswith(("caught_stealing_", "pickoff_caught_stealing_")))

        res = play.get("result", {})
        outs_after = play["count"]["outs"]
        new_home = res.get("homeScore", home_score)
        new_away = res.get("awayScore", away_score)
        post = _bases_after(play)
        desc = (res.get("description") or "").lower()
        pa_rows.append(dict(pre, event_type=res.get("eventType"),
                            outs_after=outs_after,
                            post_1b=post[0], post_2b=post[1], post_3b=post[2],
                            runs_on_play=(new_home - home_score) + (new_away - away_score),
                            final_balls=last_count["balls"], final_strikes=last_count["strikes"],
                            is_bunt=("bunt" in desc),
                            # False when the half ended on a baserunning out
                            # and the batter's plate appearance did not finish
                            pa_complete=not (res.get("eventType") or "").startswith(NON_PA_PREFIXES)))
        home_score, away_score = new_home, new_away
        outs = outs_after
        bases = post

    if (home_score, away_score) != (final_home, final_away):
        raise StateError(f"{game_pk}: replayed score {home_score}-{away_score} "
                         f"!= final {final_home}-{final_away}")
    _check_invariants(pa_rows, game_pk)
    box = feed["liveData"]["boxscore"]["teams"]
    box_sb = sum(int(box[t]["teamStats"]["batting"].get("stolenBases", 0)) for t in ("home", "away"))
    box_cs = sum(int(box[t]["teamStats"]["batting"].get("caughtStealing", 0)) for t in ("home", "away"))
    if (box_sb, box_cs) != tuple(steal_counts):
        raise StateError(f"{game_pk}: extracted SB/CS {steal_counts} != box score {box_sb}/{box_cs}")
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
            if (r["details"].get("eventType") or "").startswith("caught_stealing_"):
                for c in r.get("credits", []):
                    if c["position"]["abbreviation"] == "C":
                        credited[(p["atBatIndex"], r["details"].get("playIndex"))] = c["player"]["id"]
    rows = {(row["at_bat_index"], row["event_index"]): row for row in run_rows
            if row["event_type"].startswith("caught_stealing_")}
    missing = set(credited) - set(rows)
    if missing:
        raise StateError(f"{game_pk}: caught stealing credited but not extracted: {sorted(missing)}")
    for key, cid in credited.items():
        if rows[key]["catcher"] != cid:
            raise StateError(f"{game_pk}: tracked catcher {rows[key]['catcher']} != credited {cid}")


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

    sb_cs = [r for r in run if r["row_kind"] == "outcome"
             and r["event_type"].startswith(("stolen_base_2b", "stolen_base_3b",
                                             "caught_stealing_2b", "caught_stealing_3b"))]
    share = sum(r["on_pitch"] for r in sb_cs) / max(len(sb_cs), 1)
    print(f"{season}: steals/caught stealing of 2B/3B {len(sb_cs)}, on a pitch {share:.3f}", flush=True)
    if share < 0.90:
        sys.exit(f"{season}: only {share:.1%} of steals placed on a pitch - not writing")
    cs = [r for r in sb_cs if r["event_type"].startswith("caught_stealing_")]
    on_no_c = sum(1 for r in cs if r["on_pitch"] and "C" not in r["credited_positions"].split(","))
    off_with_c = sum(1 for r in cs if not r["on_pitch"] and "C" in r["credited_positions"].split(","))
    print(f"{season}: caught stealing {len(cs)}: on a pitch without a catcher credit {on_no_c}, "
          f"off a pitch with a catcher credit {off_with_c}", flush=True)
    if (on_no_c + off_with_c) > 0.05 * max(len(cs), 1):
        sys.exit(f"{season}: on-pitch placement disagrees with catcher credits - not writing")

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
