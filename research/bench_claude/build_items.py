"""
Items for the bench-call study (PREREG.md): every pitching change from the
6th inning on and every pinch hitter, in a seeded sample of 2026 regular
season games played 2026-07-01..2026-09-27 (pilot: games of 2026-06-01..06-30).

  python research/bench_claude/build_items.py --states S --feeds F --out research/bench_claude/inputs --set test|pilot

For each item, built only from what was known before the play (bench_replay):
  candidates   : relievers not yet used (pitching change) / bench players not yet used (pinch hitter),
                 as the bench screen lists them
  actual       : who the manager sent in (the label; not shown to Claude)
  model_pick   : the player layer's best candidate (bench.py with the late-inning usage bullpen)
  usage_pick   : pitching change only: the available reliever the team used most in that inning and
                 lead state over the previous 30 days (pen_usage.py weights; inning 6 uses innings 7-9)
  payload_B    : what Claude sees without the model's numbers
  payload_C    : the same plus the model's value of every candidate
Items whose actual player is not among the candidates are kept with excluded=reason.
Feeds are cached in F (not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench_live as BL  # noqa: E402
import bench_replay as RP  # noqa: E402
import pen_usage as PU  # noqa: E402

SEED = 20261028
N_GAMES = {"test": 80, "pilot": 4}
WINDOW = {"test": ("2026-07-01", "2026-09-27"), "pilot": ("2026-06-01", "2026-06-30")}
C = ["K", "BB", "1B", "2B", "3B", "HR", "OUT"]


def pick_games(states: Path, which: str) -> list[int]:
    pa = pd.read_parquet(states / "pa_2026.parquet", columns=["game_pk", "date"])
    g = pa.drop_duplicates("game_pk")
    lo, hi = WINDOW[which]
    g = g[(g.date.astype(str) >= lo) & (g.date.astype(str) <= hi)].sort_values("game_pk")
    rng = np.random.default_rng(SEED + (0 if which == "test" else 1))
    return sorted(int(x) for x in rng.choice(g.game_pk.to_numpy(), size=N_GAMES[which], replace=False))


def feed(feeds: Path, pk: int) -> dict | None:
    f = feeds / f"{pk}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(4):
        got = BL.get_feed(pk)
        if got:
            f.write_text(json.dumps(got))
            time.sleep(0.3)
            return got
        time.sleep(2 ** attempt)
    return None


def rates(c) -> dict:
    c = np.asarray(c, float)
    n = float(c.sum())
    if n <= 0:
        return {"PA": 0}
    hit = c[2:6].sum()
    return {"PA": int(round(n)), "K%": round(100 * c[0] / n, 1), "BB%": round(100 * c[1] / n, 1),
            "hit%": round(100 * hit / n, 1), "XBH%": round(100 * (c[3] + c[4] + c[5]) / n, 1),
            "HR%": round(100 * c[5] / n, 1)}


def build(feeds: Path, pks: list[int]) -> list[dict]:
    items = []
    for gi, pk in enumerate(pks):
        fd = feed(feeds, pk)
        if not fd:
            print("no feed", pk, flush=True)
            continue
        gd = fd["gameData"]
        s = BL._snapshot(BL.snapshot_for_season(2026))
        names = {int(p["id"]): p.get("fullName", str(p["id"])) for p in gd.get("players", {}).values() if "id" in p}
        teams = {"home": gd["teams"]["home"], "away": gd["teams"]["away"]}
        day = date.fromisoformat(gd["datetime"]["officialDate"])
        tl = RP.timeline(fd)
        rel = PU._load(2026)[1]
        for t in tl:
            for dec in t["decisions"]:
                txt = (dec.get("text") or "").lower()
                if dec["type"] == "pitch_change" and t["inning"] >= 6:
                    task = "pitching_change"
                elif dec["type"] == "pinch" and (dec.get("pos") == "PH" or "pinch-hitter" in txt):
                    task = "pinch_hitter"
                else:
                    continue
                got = RP.moment_at(fd, t["k"])
                if got is None:
                    continue
                m, data, ex = got
                side = RP.deciding_side(t, dec)
                bat_side = "home" if m.half == "bottom" else "away"
                item = {"id": f"{pk}-{t['k']}-{task}-{dec.get('player')}", "game_pk": pk, "k": t["k"], "task": task,
                        "date": day.isoformat(), "side": side, "actual": int(dec["player"]) if dec.get("player") else None,
                        "inning": m.inning, "half": m.half, "outs": m.outs, "diff_home": m.diff}
                b = BL._bench(m.extra_runner)
                if task == "pitching_change":
                    cands = sorted(int(p) for p in ex["bullpen"])          # id order: no information in the order
                    opts = b.options(m, data, relievers=cands)
                    pool = [o for o in opts if o["option"].startswith("pitch_change:")]
                    val = {int(o["option"].split(":")[1]): -100 * o["delta_wp_offense"] for o in pool}   # gain for the fielding team
                else:
                    cands = sorted(int(p) for p in ex["bench"])
                    opts = b.options(m, data, pinch_hitters=cands)
                    pool = [o for o in opts if o["option"].startswith("pinch_hit:")]
                    val = {int(o["option"].split(":")[1]): 100 * o["delta_wp_offense"] for o in pool}
                item["candidates"] = cands
                if item["actual"] not in cands:
                    item["excluded"] = "actual_not_in_candidates"
                if len(cands) < 2:
                    item["excluded"] = item.get("excluded") or "fewer_than_2_candidates"
                item["model_value"] = {str(p): round(v, 4) for p, v in val.items()}
                item["model_pick"] = max(val, key=val.get) if val else None
                # usage pick (pitching changes): the most-used available reliever in that inning and lead state
                usage_pick = None
                if task == "pitching_change":
                    fld_lead = m.diff if m.half == "top" else -m.diff
                    ls = PU.lead_state(int(fld_lead))
                    u = PU.usage(int(teams[side]["id"]), day)
                    av = {p: u[p] for p in cands if p in u}
                    W, _ = PU.weights(av, list(av), "B")
                    ii = 3 if (m.inning >= 10 or m.inning == 6) else m.inning - 7
                    w = W.get((ii, ls), {})
                    usage_pick = max(w, key=w.get) if w else None
                item["usage_pick"] = usage_pick
                # what Claude sees
                fld_team = teams["home" if m.half == "top" else "away"]["name"]
                bat_team = teams[bat_side]["name"]
                hands = data.hands or {}
                runners = {b_: (names.get(r, str(r)) if (r and m.bases[i]) else None)
                           for i, (b_, r) in enumerate(zip(("1B", "2B", "3B"), m.runner_ids))}
                due = [m.lineup[(m.slot + j) % 9] for j in range(4)]
                common = {
                    "date": day.isoformat(), "home_team": teams["home"]["name"], "away_team": teams["away"]["name"],
                    "inning": m.inning, "half": m.half, "outs": m.outs, "runners": runners,
                }
                prev = tl[t["k"] - 1] if t["k"] else None
                hs, as_ = (prev["home_score"], prev["away_score"]) if prev else (0, 0)
                common["score"] = {"home": hs, "away": as_}
                common["batting_team"], common["fielding_team"] = bat_team, fld_team
                common["your_team"] = teams[side]["name"]
                common["records_note"] = "Player records are MLB totals 2021-2025 (nothing from 2026)."
                bat_rec = lambda p: {"id": int(p), "name": names.get(p, str(p)), "bats": hands.get(p, (None, None))[0],
                                     **rates(s["_bat_all"].get(p, np.zeros(7)))}
                pit_rec = lambda p: {"id": int(p), "name": names.get(p, str(p)), "throws": hands.get(p, (None, None))[1],
                                     **rates(s["_pit_all"].get(p, np.zeros(7)))}
                if task == "pitching_change":
                    u = PU.usage(int(teams[side]["id"]), day)
                    y1 = rel.get((int(teams[side]["id"]), day - timedelta(days=1)), set())
                    y2 = rel.get((int(teams[side]["id"]), day - timedelta(days=2)), set())
                    cand_rows = []
                    for p in cands:
                        uu = u.get(p, np.zeros((3, 4)))
                        cand_rows.append({**pit_rec(p), "last_30_days_relief_batters_faced": {
                            "7th": int(uu[0].sum()), "8th": int(uu[1].sum()), "9th": int(uu[2].sum()),
                            "innings_7_to_9_when_team_led_by_1_to_3": int(uu[:, 2].sum())},
                            "pitched_yesterday": p in y1, "pitched_each_of_last_two_days": p in y1 and p in y2})
                    payload = {**common, "task": "Your manager has decided to change pitchers now, before this plate appearance. "
                                                 "Choose who comes in from the available relievers.",
                               "pitcher_leaving": pit_rec(m.pitcher), "opponent_due_up": [bat_rec(p) for p in due],
                               "available_relievers": cand_rows}
                else:
                    payload = {**common, "task": f"Your manager has decided to send a pinch hitter now for "
                                                 f"{names.get(m.lineup[m.slot], '')}. Choose who bats from the available bench players.",
                               "batter_being_replaced": bat_rec(m.lineup[m.slot]), "opposing_pitcher": pit_rec(m.pitcher),
                               "on_deck_after": [bat_rec(p) for p in due[1:]],
                               "available_bench_players": [bat_rec(p) for p in cands]}
                item["payload_B"] = payload
                pc = json.loads(json.dumps(payload))
                key = "available_relievers" if task == "pitching_change" else "available_bench_players"
                ranked = sorted(val, key=lambda p: -val[p])
                for row in pc[key]:
                    v = val.get(row["id"])
                    row["model"] = None if v is None else {"win_prob_change_for_your_team_pts": round(v, 2),
                                                           "rank": ranked.index(row["id"]) + 1}
                pc["model_note"] = ("model = this app's win-probability model (batter x pitcher rates, park, platoon, "
                                    "late-inning bullpen usage): change in your team's win probability, in percentage "
                                    "points, if this player comes in, compared with no change; rank 1 = best by the model.")
                item["payload_C"] = pc
                items.append(item)
        print(gi + 1, pk, len(items), flush=True)
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--feeds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", choices=["test", "pilot"], required=True)
    a = ap.parse_args()
    a.feeds.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pks = pick_games(a.states, a.set)
    items = build(a.feeds, pks)
    (a.out / f"items_{a.set}.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items))
    (a.out / f"games_{a.set}.json").write_text(json.dumps(pks))
    n_ex = sum(1 for i in items if i.get("excluded"))
    print(a.set, "games", len(pks), "items", len(items), "excluded", n_ex,
          {t: sum(1 for i in items if i["task"] == t and not i.get("excluded")) for t in ("pitching_change", "pinch_hitter")})


if __name__ == "__main__":
    main()
