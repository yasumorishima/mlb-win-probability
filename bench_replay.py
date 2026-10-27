"""
Looking back at a finished game: the bench's view just before any plate
appearance, as it stood then.

  from bench_replay import timeline, moment_at
  plays = timeline(feed)                 # one entry per plate appearance
  got = moment_at(feed, k)               # (Moment, Data, extra) before play k

Rebuilt from the game feed only:
  - lineups as they stood: the starters (battingOrder "N00"), then every
    substitution (offensive / defensive) recorded in plays before k;
  - the batter due up (the slot after the team's last completed plate
    appearance), so a pinch hitter sent up in play k is an *option* at k,
    and what the manager actually did is read from play k's events;
  - pitchers used before k (the fielding team's current one, the other
    team's latest), bench players and bullpen arms not yet used;
  - player records that end with the previous season
    (bench_snapshot_<season - 1>.json): no hindsight from the game's season.
"""

from __future__ import annotations

from functools import lru_cache

import bench_live as BL
from live_feed import _game_rules, parse_game_plays

SUB_TYPES = {"offensive_substitution", "defensive_substitution", "defensive_switch"}


def _subs(play: dict) -> list[dict]:
    out = []
    for e in play.get("playEvents", []):
        d = e.get("details", {})
        if d.get("eventType") in SUB_TYPES | {"pitching_substitution"}:
            out.append({"type": d.get("eventType"), "player": (e.get("player") or {}).get("id"),
                        "order": e.get("battingOrder"), "pos": (e.get("position") or {}).get("abbreviation"),
                        "text": d.get("description", "")})
    return out


def _side(play: dict) -> tuple[str, str]:
    return ("away", "home") if play["about"]["isTopInning"] else ("home", "away")


def timeline(feed: dict) -> list[dict]:
    """Per plate appearance: the state before it (parse_game_plays), what
    happened, and what the bench did in it (substitutions, steals, bunts,
    walks)."""
    states = parse_game_plays(feed)
    raw = [p for p in feed.get("liveData", {}).get("plays", {}).get("allPlays", []) if p.get("about", {}).get("isComplete")]
    out = []
    for k, (st, p) in enumerate(zip(states, raw)):
        res = p.get("result", {})
        events = [e.get("details", {}) for e in p.get("playEvents", [])]
        decisions = []
        for sb in _subs(p):
            decisions.append({"type": {"offensive_substitution": "pinch", "pitching_substitution": "pitch_change"}.get(sb["type"], "defense"),
                              "player": sb["player"], "pos": sb["pos"], "text": sb["text"]})
        for d in events:
            et = d.get("eventType") or ""
            if et.startswith(("stolen_base", "caught_stealing")):
                decisions.append({"type": "steal", "text": d.get("description", ""), "ok": et.startswith("stolen_base")})
        if res.get("eventType") == "intent_walk":
            decisions.append({"type": "intentional_walk", "text": res.get("description", "")})
        if "bunt" in (res.get("description") or "").lower():
            decisions.append({"type": "bunt", "text": res.get("description", "")})
        out.append({**st, "k": k, "decisions": decisions, "event": res.get("event", ""),
                    "description": res.get("description", "")})
    return out


def moment_at(feed: dict, k: int, snapshot_path: str | None = "auto"):
    """(Moment, Data, extra) just before plate appearance k, or None."""
    raw = [p for p in feed.get("liveData", {}).get("plays", {}).get("allPlays", []) if p.get("about", {}).get("isComplete")]
    states = parse_game_plays(feed)
    if not 0 <= k < len(raw):
        return None
    gd = feed.get("gameData", {})
    box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    season = int(gd.get("game", {}).get("season") or str(gd.get("datetime", {}).get("officialDate", "2026"))[:4])
    path = BL.snapshot_for_season(season) if snapshot_path == "auto" else snapshot_path
    s = BL._snapshot(path)

    # lineups, used players and pitchers before play k
    lineup, used, pitchers = {}, {}, {}
    for side in ("home", "away"):
        lu = [None] * 9
        for p in box.get(side, {}).get("players", {}).values():
            bo = str(p.get("battingOrder", ""))
            if bo.endswith("00") and len(bo) == 3:
                lu[int(bo[0]) - 1] = int(p["person"]["id"])
        lineup[side], used[side], pitchers[side] = lu, set(x for x in lu if x), []
    n_pa = {"home": 0, "away": 0}
    for j in range(k):
        p = raw[j]
        bat, fld = _side(p)
        for sb in _subs(p):
            side = bat if sb["type"] == "offensive_substitution" else fld
            if sb["type"] == "pitching_substitution":
                continue
            if sb["order"] and sb["player"]:
                lineup[side][int(str(sb["order"])[0]) - 1] = int(sb["player"])
            if sb["player"]:
                used[side].add(int(sb["player"]))
        pid = p.get("matchup", {}).get("pitcher", {}).get("id")
        if pid and (not pitchers[fld] or pitchers[fld][-1] != pid):
            pitchers[fld].append(int(pid))
        if p.get("result", {}).get("type") == "atBat":
            n_pa[bat] += 1
    p = raw[k]
    bat, fld = _side(p)
    if any(x is None for x in lineup[bat] + lineup[fld]):
        return None
    slot, fld_next = n_pa[bat] % 9, n_pa[fld] % 9
    batter = lineup[bat][slot]
    # the fielding team's pitcher before any change made in play k
    if not pitchers[fld]:
        first = next((q for q in raw if _side(q)[1] == fld), None)
        pitchers[fld].append(int(first["matchup"]["pitcher"]["id"]))
    if not pitchers[bat]:
        first = next((q for q in raw if _side(q)[1] == bat), None)
        if first is None:
            return None
        pitchers[bat].append(int(first["matchup"]["pitcher"]["id"]))

    def roster(side, want_pitchers):
        out = []
        for q in box.get(side, {}).get("players", {}).values():
            is_p = (q.get("position") or {}).get("abbreviation") == "P"
            if is_p == want_pitchers:
                out.append(int(q["person"]["id"]))
        return out

    bench = [x for x in roster(bat, False) if x not in used[bat]]
    pen_fld = [x for x in roster(fld, True) if x not in pitchers[fld]]
    pen_bat = [x for x in roster(bat, True) if x not in pitchers[bat]]
    st = dict(states[k])
    st["outs"] = st["outs"]
    prev = raw[k - 1] if k else None
    same_half = prev is not None and prev["about"]["inning"] == p["about"]["inning"] and \
        prev["about"]["isTopInning"] == p["about"]["isTopInning"]
    m = prev.get("matchup", {}) if same_half else {}
    runner_ids = tuple((m.get(key) or {}).get("id") for key in ("postOnFirst", "postOnSecond", "postOnThird"))
    sched, xr = _game_rules(feed)
    st.update({"scheduled_innings": sched, "extra_runner": xr})
    got = BL.assemble(feed, st, s, half=st["top_bottom"], lineup=lineup[bat], slot=slot, batter=batter,
                      fld_lineup=lineup[fld], fld_next=fld_next, pitcher=pitchers[fld][-1], bat_pitcher=pitchers[bat][-1],
                      cur_is_starter=len(pitchers[fld]) == 1, bat_pitcher_is_starter=len(pitchers[bat]) == 1,
                      bench=bench, bullpen_fld=pen_fld, bullpen_bat=pen_bat,
                      fld_team=box.get(fld, {}).get("team", {}).get("id"), bat_team=box.get(bat, {}).get("team", {}).get("id"),
                      runner_ids=runner_ids)
    got[2]["season"] = season
    return got


# --- rating the manager's calls --------------------------------------------

OFFENSE_DECISIONS = {"pinch", "steal", "bunt"}
KINDS = {"pinch": ("pinch_hit", "pinch_run"), "pitch_change": ("pitch_change",)}


def deciding_side(t: dict, dec: dict) -> str:
    """'home' / 'away': the team that made the call."""
    bat = "home" if t["top_bottom"] == "bottom" else "away"
    return bat if dec["type"] in OFFENSE_DECISIONS else ("away" if bat == "home" else "home")


def rate(dec: dict, opts: dict) -> dict | None:
    """How the model rated a call before the play, from the deciding team's
    side: gain = its WP with the call minus its WP with no move (points),
    best = the best candidate of the same kind, rank among those candidates.
    None if the call is not among the options (e.g. a player without records)."""
    allo = opts["all"]
    sign = 1 if dec["type"] in OFFENSE_DECISIONS else -1
    if dec["type"] == "steal":
        cands = [o for o in allo if o["kind"] == "steal"]
        if not cands:
            return None
        txt = dec.get("text", "")
        me = next((o for o in cands if ("3rd" in txt) == (o["option"].startswith("steal_3b"))), cands[0])
        return {"gain": 100 * me["delta_wp_offense"], "call": "go" if me["go"] else "hold", "me": me,
                "p_success": me["p_success"], "breakeven": me["breakeven"], "agree": bool(me["go"]),
                "rank": None, "n": None, "best": None, "best_gain": None}
    if dec["type"] in ("bunt", "intentional_walk"):
        me = next((o for o in allo if o["kind"] == dec["type"]), None)
        if me is None:
            return None
        g = sign * 100 * me["delta_wp_offense"]
        return {"gain": g, "agree": g > 0, "me": me, "rank": None, "n": None, "best": None, "best_gain": None, "call": None}
    kinds = KINDS.get(dec["type"])
    if not kinds:
        return None
    if dec["type"] == "pinch":
        txt = (dec.get("text") or "").lower()
        if dec.get("pos") == "PR" or "pinch-runner" in txt:
            kinds = ("pinch_run",)
        elif dec.get("pos") == "PH" or "pinch-hitter" in txt:
            kinds = ("pinch_hit",)
    me = next((o for o in allo if o["kind"] in kinds and o["option"].endswith(f":{dec.get('player')}")), None)
    if me is None:
        return None
    # rank among candidates of the same kind (a pinch runner against pinch runners)
    cands = sorted([o for o in allo if o["kind"] == me["kind"]], key=lambda o: -sign * o["wp_offense"])
    best = cands[0]
    return {"gain": sign * 100 * me["delta_wp_offense"], "rank": cands.index(me) + 1, "n": len(cands),
            "best": best, "best_gain": sign * 100 * best["delta_wp_offense"], "agree": cands.index(me) == 0, "me": me,
            "call": None}


def game_review(feed: dict, series: list[dict] | None = None, snapshot_path: str | None = "auto") -> list[dict]:
    """Every call of the game (substitutions in the field excluded) with the
    model's rating before the play and the WP swing over the play.
    `series`: bench_viz.wp_series(feed) (home WP before play i at index i)."""
    tl = timeline(feed)
    rows = []
    for t in tl:
        decs = [d for d in t["decisions"] if d["type"] != "defense"]
        if not decs:
            continue
        got = moment_at(feed, t["k"], snapshot_path)
        opts = BL.bench_options(feed, got=got) if got is not None else None
        for d in decs:
            side = deciding_side(t, d)
            r = rate(d, opts) if opts else None
            sw = None
            if series and t["k"] + 1 < len(series):
                a, b = series[t["k"]]["wp"], series[t["k"] + 1]["wp"]
                sw = 100 * ((b - a) if side == "home" else (a - b))
            rows.append({"k": t["k"], "inning": t["inning"], "top_bottom": t["top_bottom"], "outs": t["outs"],
                         "runners": t["runners"], "side": side, "type": d["type"], "text": d.get("text", ""),
                         "rating": r, "swing": sw, "event": t["event"]})
    return rows


MISSED_KINDS = {"pinch_hit": "pinch", "pinch_run": "pinch", "bunt": "bunt", "steal": "steal",
                "intentional_walk": "intentional_walk", "pitch_change": "pitch_change"}


def missed_chances(feed: dict, min_gain: float = 1.0, top: int = 5, pitch_from: int = 7,
                   snapshot_path: str | None = "auto", progress=None) -> list[dict]:
    """Moments where the model's best move gained at least `min_gain` WP
    points for one team and that team made no call of that kind in the play.
    Pitching changes only from inning `pitch_from` (earlier ones are
    overstated: the model's bullpen after the change is the pooled average)."""
    tl = timeline(feed)
    out = []
    for i, t in enumerate(tl):
        if progress:
            progress(i / max(len(tl), 1))
        got = moment_at(feed, t["k"], snapshot_path)
        if got is None:
            continue
        opts = BL.bench_options(feed, got=got)
        made = {(deciding_side(t, d), d["type"]) for d in t["decisions"]}
        bat = "home" if t["top_bottom"] == "bottom" else "away"
        fld = "away" if bat == "home" else "home"
        best = None
        for o in opts["all"]:
            kind = o["kind"]
            if kind not in MISSED_KINDS or (kind == "pitch_change" and t["inning"] < pitch_from):
                continue
            if kind == "steal" and not o["go"]:
                continue
            side = bat if o["side"] == "offense" else fld
            g = 100 * o["delta_wp_offense"] * (1 if o["side"] == "offense" else -1)
            if g < min_gain or (side, MISSED_KINDS[kind]) in made:
                continue
            if best is None or g > best["gain"]:
                best = {"k": t["k"], "inning": t["inning"], "top_bottom": t["top_bottom"], "outs": t["outs"],
                        "runners": t["runners"], "batter": t["batter"], "side": side, "option": o, "gain": g}
        if best:
            out.append(best)
    if progress:
        progress(1.0)
    return sorted(out, key=lambda r: -r["gain"])[:top]
