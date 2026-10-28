"""
Bench options for a live game: who to send up, whether to bunt or walk,
who to bring in — each scored with the player layer
(research/player_layer, pre-registered and confirmed on 2026:
research/player_layer/RESULTS_2026.md).

    from bench_live import bench_options
    out = bench_options(feed)        # feed = /api/v1.1/game/{pk}/feed/live JSON

Inputs come from the live feed (lineups as they stand, bench, available
bullpen arms, current pitcher, next batter of the other team) and from a
snapshot of player records (`research/player_layer/published/bench_snapshot.json`,
2021-2026 regular seasons; not updated during a later season — say so on
screen). The pooled bullpen is the arms the feed lists as available.

Not modelled: pinch runners (base running is league average), a reliever
beyond this half-inning, the on-deck batter being walked around, defense
of individual fielders (no measurable effect in the sizing).
"""

from __future__ import annotations

import json
from datetime import date
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research" / "player_layer"))
import bench as B  # noqa: E402
import pen_usage as PU  # noqa: E402
from live_feed import _BASE, _fetch, parse_live_state  # noqa: E402
from wp_engine import Engine  # noqa: E402

SNAPSHOT = ROOT / "research/player_layer/published/bench_snapshot.json"
MAX_PINCH, MAX_RELIEF = 6, 6

LABEL_JA = {"none": "そのまま", "bunt": "送りバント", "intentional_walk": "敬遠"}
LABEL_EN = {"none": "no move", "bunt": "sac bunt", "intentional_walk": "intentional walk"}
MAX_RUN = 3


@lru_cache(maxsize=8)
def _snapshot(path: str | None = None) -> dict:
    s = json.loads((Path(path) if path else SNAPSHOT).read_text())
    arr = lambda d: {int(k): np.asarray(v, float) for k, v in d.items()}
    s["_bat_all"], s["_pit_all"] = arr(s["bat"]), arr(s["pit"])
    # season-decayed records (size_decay.py); the bullpen pool and roles use them too
    s["_bat"], s["_pit"] = arr(s.get("bat_decay", s["bat"])), arr(s.get("pit_decay", s["pit"]))
    s["_bat_vs"] = {(int(k.split("|")[0]), k.split("|")[1]): np.asarray(v, float) for k, v in s["bat_vs"].items()}
    s["_pit_vs"] = {(int(k.split("|")[0]), k.split("|")[1]): np.asarray(v, float) for k, v in s["pit_vs"].items()}
    s["_ct"] = {h: {"P": np.asarray(v["P"]), "T": np.asarray(v["T"])} for h, v in s["class_tables"].items()}
    s["_hands"] = {int(k): tuple(v) for k, v in s["hands"].items()}
    return s


def snapshot_for_season(season: int) -> str | None:
    """Records known before a game of `season`: the snapshot that ends with
    the previous season (looking back at past games without hindsight);
    the latest snapshot for the current season."""
    f = SNAPSHOT.parent / f"bench_snapshot_{season - 1}.json"
    return str(f) if f.exists() else None


@lru_cache(maxsize=2)
def _bench(extra_runner: bool = True) -> B.Bench:
    return B.Bench(Engine.load(ROOT / "engine_tables.json", extra_runner=extra_runner), _snapshot()["_ct"])


def get_feed(game_pk: int) -> dict | None:
    return _fetch(f"{_BASE}/v1.1/game/{game_pk}/feed/live")


def _ids(xs) -> list[int]:
    return [int(x) for x in xs or []]


def _slot(order: list[int], pid) -> int | None:
    try:
        return order.index(int(pid))
    except (ValueError, TypeError):
        return None


def _hand_from_feed(feed: dict, hands: dict) -> dict:
    """Add bat side / pitch hand of players the snapshot does not know."""
    out = dict(hands)
    for pid, p in feed.get("gameData", {}).get("players", {}).items():
        i = int(p.get("id", 0))
        if i and i not in out:
            out[i] = (p.get("batSide", {}).get("code"), p.get("pitchHand", {}).get("code"))
    return out


def moment_from_feed(feed: dict) -> tuple[B.Moment, B.Data, dict] | None:
    """The bench's view of the current moment, or None if it cannot be read
    (game not in progress, lineups missing)."""
    st = parse_live_state(feed)
    if st["outs"] >= 3:
        return None
    s = _snapshot()
    gd, ld = feed.get("gameData", {}), feed.get("liveData", {})
    box, ls = ld.get("boxscore", {}).get("teams", {}), ld.get("linescore", {})
    half = st["top_bottom"]
    bat_side, fld_side = ("home", "away") if half == "bottom" else ("away", "home")
    bat_box, fld_box = box.get(bat_side, {}), box.get(fld_side, {})
    lineup, fld_lineup = _ids(bat_box.get("battingOrder")), _ids(fld_box.get("battingOrder"))
    if len(lineup) != 9 or len(fld_lineup) != 9:
        return None
    bat_id = bat_box.get("team", {}).get("id")
    off, dfn_ = ls.get("offense", {}), ls.get("defense", {})
    # the linescore's "offense" is the team at bat now; between halves it may still be the other one
    if off.get("team", {}).get("id") == bat_id:
        batter, fld_next_id = off.get("batter", {}).get("id"), dfn_.get("batter", {}).get("id")
    else:
        batter, fld_next_id = dfn_.get("batter", {}).get("id"), off.get("batter", {}).get("id")
    slot, fld_next = _slot(lineup, batter), _slot(fld_lineup, fld_next_id)
    if slot is None:
        return None
    fld_pitchers, bat_pitchers = _ids(fld_box.get("pitchers")), _ids(bat_box.get("pitchers"))
    pitcher = fld_pitchers[-1] if fld_pitchers else None
    if pitcher is None:
        prob = gd.get("probablePitchers", {}).get(fld_side, {})
        pitcher = prob.get("id")
    bp = bat_pitchers[-1] if bat_pitchers else gd.get("probablePitchers", {}).get(bat_side, {}).get("id")
    if pitcher is None or bp is None:
        return None
    fld_pitchers, bat_pitchers = _ids(fld_box.get("pitchers")), _ids(bat_box.get("pitchers"))
    runner_ids = tuple((off.get(b) or {}).get("id") if off.get("team", {}).get("id") == bat_id else None
                       for b in ("first", "second", "third"))
    return assemble(feed, st, s, half=half, lineup=lineup, slot=slot, batter=int(batter), fld_lineup=fld_lineup,
                    fld_next=fld_next, pitcher=int(pitcher), bat_pitcher=int(bp),
                    cur_is_starter=len(fld_pitchers) <= 1, bat_pitcher_is_starter=len(bat_pitchers) <= 1,
                    bench=_ids(bat_box.get("bench")), bullpen_fld=_ids(fld_box.get("bullpen")),
                    bullpen_bat=_ids(bat_box.get("bullpen")), fld_team=fld_box.get("team", {}).get("id"),
                    bat_team=bat_id, runner_ids=runner_ids)


def pen_usage_for(gd: dict, team, bullpen) -> dict | None:
    """Available relievers' late-inning usage over the 60 days before the
    game (pen_usage.py); None when there is no usage table for the season
    (e.g. a live game after the last published season): pooled bullpen."""
    day = str(gd.get("datetime", {}).get("officialDate") or "")
    if not day or team is None or gd.get("game", {}).get("type", "R") != "R":
        return None
    d = date.fromisoformat(day[:10])
    if not PU.available(d.year):
        return None
    u, tired = PU.usage(int(team), d, window=PU.BENCH_WINDOW_DAYS), PU.tired(int(team), d)
    roster = {int(i) for i in bullpen or []}
    return {p: v for p, v in u.items() if p not in tired and (not roster or p in roster)}


def assemble(feed: dict, st: dict, s: dict, *, half, lineup, slot, batter, fld_lineup, fld_next, pitcher, bat_pitcher,
             cur_is_starter, bat_pitcher_is_starter, bench, bullpen_fld, bullpen_bat, fld_team, bat_team, runner_ids):
    """Moment + Data + extra from the pieces of a game moment (live or replayed)."""
    gd = feed.get("gameData", {})
    home_abbr = s["team_abbr"].get(str(gd.get("teams", {}).get("home", {}).get("id")))
    park = s["park"].get(home_abbr)
    L = np.asarray(s["L"])
    zeros = np.zeros(len(L))

    def pool(ids):
        tot = sum((s["_pit"].get(i, zeros) for i in ids), zeros.copy())
        return tot if tot.sum() > 0 else L * 2000          # unknown arms: league average

    m = B.Moment(inning=st["inning"], half=half, outs=st["outs"], bases=tuple(st["runners"]),
                 diff=st["score_diff"], sched=st["scheduled_innings"], lineup=lineup, slot=slot,
                 pitcher=int(pitcher), fld_lineup=fld_lineup, fld_next=fld_next if fld_next is not None else 0,
                 bat_pitcher=int(bat_pitcher), cur_is_starter=cur_is_starter,
                 bat_pitcher_is_starter=bat_pitcher_is_starter,
                 park=np.asarray(park) if park else None, extra_runner=bool(st.get("extra_runner", True)),
                 runner_ids=tuple(runner_ids))
    dec = s.get("decay")
    prm = None
    if dec and "bat_decay" in s:
        import layer as Ly
        prm = {k: (v * dec["k_scale"] if k != "kd" else v) for k, v in Ly.PARAMS.items()}
    data = B.Data(L=L, bat=s["_bat"], pit=s["_pit"], prm=prm,
                  speed={int(k): v for k, v in s.get("speed", {}).items()}, speed_coef=s.get("speed_coef", {}),
                  pen_fld=pool(bullpen_fld), pen_bat=pool(bullpen_bat),
                  def_fld=np.asarray(s["dfn"].get(str(fld_team), [0, 0, 0, 0]), float),
                  def_bat=np.asarray(s["dfn"].get(str(bat_team), [0, 0, 0, 0]), float),
                  hands=_hand_from_feed(feed, s["_hands"]), split_bat=s["_bat_vs"], split_pit=s["_pit_vs"],
                  platoon={k: np.asarray(v) for k, v in s["platoon"].items()})
    data.pen_avail_fld, data.pen_avail_bat = pen_usage_for(gd, fld_team, bullpen_fld), pen_usage_for(gd, bat_team, bullpen_bat)
    names = {int(p["id"]): p.get("fullName", str(p["id"])) for p in gd.get("players", {}).values() if "id" in p}
    role = s.get("role", {})
    relievers = [i for i in bullpen_fld
                 if (r := role.get(str(i))) is None or r[0] * 2 < r[1]]   # mostly relief work in the last two seasons (unknown: kept)
    extra = {"bench": list(bench), "bullpen": relievers, "names": names,
             "state": st, "batter": int(batter), "records_through": s["records_through"],
             "pen_usage": data.pen_avail_fld is not None}
    return m, data, extra


def bench_options(feed: dict, max_pinch: int = MAX_PINCH, max_relief: int = MAX_RELIEF, got=None) -> dict | None:
    """Every option with the rest-of-half expected runs, the chance of
    scoring and the offense's WP; the best pinch hitters / relievers first.
    `got`: a prepared (Moment, Data, extra), e.g. from bench_replay.moment_at."""
    got = got or moment_from_feed(feed)
    if got is None:
        return None
    m, data, extra = got
    bench = _bench(m.extra_runner)
    opts = bench.options(m, data, pinch_hitters=extra["bench"], relievers=extra["bullpen"], pinch_runners=extra["bench"])
    nm = extra["names"]
    for o in opts:
        kind, _, pid = o["option"].partition(":")
        o["kind"] = kind
        if kind.startswith("steal_"):
            who = nm.get(int(pid), "") if pid not in ("", "None") else ""
            base = "二盗" if kind == "steal_2b" else "三盗"
            o["label_ja"] = (f"{base}（{who} 成功見込み {o['p_success']:.0%} / 必要 {o['breakeven']:.0%}）"
                             f"→ {'走れ' if o['go'] else '走るな'}")
            o["label_en"] = (f"steal {kind[-2:]} ({who} est. {o['p_success']:.0%} / needed {o['breakeven']:.0%}) "
                             f"-> {'go' if o['go'] else 'hold'}")
            o["kind"] = "steal"
            continue
        o["label_ja"] = LABEL_JA.get(kind) or {"pinch_hit": "代打", "pinch_run": "代走", "pitch_change": "継投"}[kind] + \
            f" {nm.get(int(pid), pid)}"
        o["label_en"] = LABEL_EN.get(kind) or {"pinch_hit": "PH", "pinch_run": "PR", "pitch_change": "new P"}[kind] + \
            f" {nm.get(int(pid), pid)}"
    ph = sorted([o for o in opts if o["kind"] == "pinch_hit"], key=lambda o: -o["wp_offense"])[:max_pinch]
    rp = sorted([o for o in opts if o["kind"] == "pitch_change"], key=lambda o: o["wp_offense"])[:max_relief]
    pr = sorted([o for o in opts if o["kind"] == "pinch_run"], key=lambda o: -o["wp_offense"])[:MAX_RUN]
    offense = [o for o in opts if o["kind"] in ("none", "bunt", "steal")] + ph + pr
    defense = [o for o in opts if o["kind"] == "intentional_walk"] + rp
    return {"offense": offense, "defense": defense, "all": opts, "batter": nm.get(extra["batter"], ""),
            "pitcher": nm.get(m.pitcher, ""), "records_through": extra["records_through"],
            "state": extra["state"], "pen_plan": pen_plan(m, data, nm)}


def pen_plan(m, data, names: dict, top: int = 3) -> list[dict] | None:
    """Who the model expects to pitch the 7th / 8th / 9th for the fielding
    team at the current lead (usage mixture without the current pitcher):
    [{inning, rows: [(name, share)], other: share of the pooled bullpen}]."""
    if data.pen_avail_fld is None:
        return None
    av = {p: u for p, u in data.pen_avail_fld.items() if p != m.pitcher}
    W, tot = PU.weights(av, list(av), "B")
    pool_w = PU.KAPPA / (PU.KAPPA + tot)
    lead = m.diff if m.half == "top" else -m.diff          # the fielding team is home in the top half
    ls = PU.lead_state(int(lead))
    out = []
    for ii, inn in enumerate((7, 8, 9)):
        if inn < m.inning:
            continue
        w = W[(ii, ls)]
        sw = sum(w.values())
        if sw <= 0:
            out.append({"inning": inn, "rows": [], "other": 1.0})
            continue
        rows = sorted(((names.get(p, str(p)), (1 - pool_w) * x / sw) for p, x in w.items()), key=lambda r: -r[1])
        out.append({"inning": inn, "rows": rows[:top], "other": 1 - sum(r[1] for r in rows[:top])})
    return out


def matchups(feed: dict, n_next: int = 3, got=None) -> dict | None:
    """The current batter and the next `n_next` against the current pitcher:
    the plate-appearance outcome probabilities of the player layer in this
    base-out state (handedness included)."""
    got = got or moment_from_feed(feed)
    if got is None:
        return None
    m, data, extra = got
    from wp_engine import STATES, _state
    ct = _snapshot()["_ct"][m.half]
    si = STATES.index(_state(m.outs, *m.bases))
    ids = [m.lineup[(m.slot + k) % 9] for k in range(n_next + 1)]
    M = B._mult(data, ids, m.pitcher, B._counts(data.pit, m.pitcher), data.def_fld, m.park)
    P = ct["P"][si]
    po = P[len(B.C)]
    q = P[None, :len(B.C)] * M
    q = q / q.sum(axis=1, keepdims=True)
    C = B.C
    nm = extra["names"]
    rows = []
    for k, (pid, p) in enumerate(zip(ids, q)):
        rows.append({"order": k, "id": pid, "name": nm.get(pid, str(pid)),
                     "p_on_base": float(p[C.index("BB")] + p[C.index("1B")] + p[C.index("2B")] + p[C.index("3B")] + p[C.index("HR")]),
                     "p_k": float(p[C.index("K")]), "p_xbh": float(p[C.index("2B")] + p[C.index("3B")] + p[C.index("HR")]),
                     "p_hr": float(p[C.index("HR")]), "p_bb": float(p[C.index("BB")]), "p_1b": float(p[C.index("1B")]),
                     "p_2b": float(p[C.index("2B")]), "p_3b": float(p[C.index("3B")]), "p_out": float(p[C.index("OUT")])})
    return {"pitcher": nm.get(m.pitcher, str(m.pitcher)), "rows": rows, "p_other": float(po)}
