"""
Per-game inputs for the look-back game notes (regular season; --season, default 2026).

  python research/game_notes/build_inputs.py --states S --out research/game_notes/work/inputs_2026.jsonl

One JSON object per game, built only from the plate-appearance table (pa_2026), the look-back win
probability (research/player_layer/published/pen_wp_2026.parquet, PENR_RECAL) and names from the
MLB Stats API (teams, people). Every number in a note must come from these objects
(check_notes.py enforces it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PEN_DIR = ROOT / "research/player_layer/published"
EVENT_JA = {"single": "シングルヒット", "double": "二塁打", "triple": "三塁打", "home_run": "ホームラン", "walk": "四球",
            "intent_walk": "敬遠", "hit_by_pitch": "死球", "strikeout": "三振", "field_out": "凡打", "force_out": "封殺",
            "grounded_into_double_play": "併殺打", "double_play": "併殺", "sac_fly": "犠牲フライ", "sac_bunt": "犠牲バント",
            "field_error": "失策", "fielders_choice": "野選", "fielders_choice_out": "野選（アウト）",
            "strikeout_double_play": "三振併殺", "catcher_interf": "打撃妨害", "sac_fly_double_play": "犠飛併殺",
            "wild_pitch": "暴投", "caught_stealing_2b": "盗塁死", "caught_stealing_3b": "盗塁死", "caught_stealing_home": "本盗死"}


def names(ids: list[int]) -> dict[int, str]:
    out = {}
    for i in range(0, len(ids), 150):
        q = ",".join(str(x) for x in ids[i:i + 150])
        for k in range(4):
            try:
                r = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={q}", timeout=30)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 ** k)
        for p in r.json().get("people", []):
            out[int(p["id"])] = p["fullName"]
    return out


def runners(row) -> str:
    on = [b for b, c in (("一", "on_1b"), ("二", "on_2b"), ("三", "on_3b")) if pd.notna(row[c])]
    return "走者なし" if not on else "満塁" if len(on) == 3 else "".join(on) + "塁"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--newest-first", action="store_true", help="write games in reverse date order")
    a = ap.parse_args()
    pa = pd.read_parquet(a.states / f"pa_{a.season}.parquet").sort_values(["game_pk", "at_bat_index"])
    wp = pd.read_parquet(PEN_DIR / f"pen_wp_{a.season}.parquet")
    pa = pa.merge(wp, on=["game_pk", "at_bat_index"], how="left")
    if pa.wp.isna().any():
        sys.exit("missing look-back WP")
    teams = {t["id"]: t["name"] for t in requests.get(f"https://statsapi.mlb.com/api/v1/teams?sportId=1&season={a.season}", timeout=30).json()["teams"]}
    nm = names(sorted(set(pa.batter.astype(int)) | set(pa.pitcher.astype(int))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.out.open("w") as f:
        order = pa.groupby("game_pk").date.first().sort_values(ascending=not a.newest_first, kind="stable").index
        groups = dict(tuple(pa.groupby("game_pk", sort=False)))
        for g in order:
            d = groups[g]
            d = d.reset_index(drop=True)
            home, away = teams[int(d.home_team[0])], teams[int(d.away_team[0])]
            last = d.iloc[-1]
            hs = int(last.home_score + (last.runs_on_play if last.half == "bottom" else 0))
            as_ = int(last.away_score + (last.runs_on_play if last.half == "top" else 0))
            home_won = int(last.home_won)
            w = d.wp.to_numpy()
            after = np.append(w[1:], float(home_won))
            swing_home = after - w
            bat_home = (d.half == "bottom").to_numpy()
            swing_bat = np.where(bat_home, swing_home, -swing_home)
            top = np.argsort(-np.abs(swing_bat))[:4]
            plays = []
            for i in sorted(top):
                r = d.iloc[i]
                bteam = home if r.half == "bottom" else away
                bw0 = w[i] if r.half == "bottom" else 1 - w[i]
                plays.append({"inning": f"{int(r.inning)}回{'裏' if r.half == 'bottom' else '表'}", "outs": int(r.outs),
                              "runners": runners(r), "score_before": f"{away} {int(r.away_score)} - {int(r.home_score)} {home}",
                              "batting_team": bteam, "batter": nm.get(int(r.batter), str(r.batter)),
                              "pitcher": nm.get(int(r.pitcher), str(r.pitcher)),
                              "result": EVENT_JA.get(r.event_type, r.event_type), "runs_scored": int(r.runs_on_play),
                              "score_after": f"{away} {int(r.away_score) + (int(r.runs_on_play) if r.half == 'top' else 0)} - "
                                             f"{int(r.home_score) + (int(r.runs_on_play) if r.half == 'bottom' else 0)} {home}",
                              "batting_team_wp_before_pct": int(round(100 * bw0)),
                              "batting_team_wp_after_pct": int(round(100 * (bw0 + swing_bat[i])))})
            changes = []
            prev = {}
            for i, r in d.iterrows():
                fld = "home" if r.half == "top" else "away"
                p = int(r.pitcher)
                if fld in prev and prev[fld] != p and int(r.inning) >= 6:
                    fw = w[i] if fld == "home" else 1 - w[i]
                    changes.append({"inning": f"{int(r.inning)}回{'裏' if r.half == 'bottom' else '表'}",
                                    "team": home if fld == "home" else away, "new_pitcher": nm.get(p, str(p)),
                                    "score": f"{away} {int(r.away_score)} - {int(r.home_score)} {home}",
                                    "team_wp_pct": int(round(100 * fw))})
                prev[fld] = p
            obj = {"game_pk": int(g), "date": str(pd.Timestamp(d.date[0]).date()), "away": away, "home": home,
                   "final": f"{away} {as_} - {hs} {home}", "winner": home if home_won else away,
                   "innings": int(d.inning.max()), "home_wp_start_pct": int(round(100 * w[0])),
                   "home_wp_min_pct": int(round(100 * w.min())), "home_wp_max_pct": int(round(100 * w.max())),
                   "big_plays": plays, "late_pitching_changes": changes[:6]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print("games", n, a.out)


if __name__ == "__main__":
    main()
