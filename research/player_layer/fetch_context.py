"""
Fetch per-game fielding alignment and player handedness from the MLB Stats
API (keyless) for the player layer. Output goes to a scratch directory
(raw Stats API data is not committed).

  python research/player_layer/fetch_context.py alignment --states <dir> --out <dir> [--seasons 2021 ...]
  python research/player_layer/fetch_context.py hands --states <dir> --out <dir>

alignment: for every game, each team's starting fielders (battingOrder
  "N00" entries, position C/1B/2B/3B/SS/LF/CF/RF; DH and pitchers left out)
  -> <out>/alignment_<season>.parquet (game_pk, team, pos, player)
hands: batSide / pitchHand of every batter and pitcher in the tables
  -> <out>/hands.parquet (player, bat_side, pitch_hand)
Resumable: games already in the output are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://statsapi.mlb.com/api/v1"
FIELD = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"}


def fetch(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlb-win-probability/2.0 (research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def alignment(states: Path, out: Path, seasons) -> None:
    for s in seasons:
        games = sorted(pd.read_parquet(states / f"pa_{s}.parquet", columns=["game_pk"]).game_pk.unique())
        f = out / f"alignment_{s}.parquet"
        have = set(pd.read_parquet(f).game_pk) if f.exists() else set()
        rows = pd.read_parquet(f).to_dict("records") if f.exists() else []
        todo = [g for g in games if g not in have]
        for i, g in enumerate(todo):
            box = fetch(f"{BASE}/game/{g}/boxscore")
            for side in ("home", "away"):
                t = box["teams"][side]
                for p in t["players"].values():
                    bo = str(p.get("battingOrder", ""))
                    pos = p.get("position", {}).get("abbreviation")
                    if bo.endswith("00") and pos in FIELD:
                        rows.append({"game_pk": int(g), "team": int(t["team"]["id"]), "pos": pos,
                                     "player": int(p["person"]["id"])})
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                pd.DataFrame(rows).to_parquet(f, index=False)
                print(s, i + 1, "/", len(todo), flush=True)
            time.sleep(0.1)


def hands(states: Path, out: Path) -> None:
    ids = set()
    for p in sorted(states.glob("pa_*.parquet")):
        d = pd.read_parquet(p, columns=["batter", "pitcher"])
        ids |= set(d.batter.astype(int)) | set(d.pitcher.astype(int))
    ids = sorted(ids)
    rows = []
    for k in range(0, len(ids), 150):
        js = fetch(f"{BASE}/people?personIds={','.join(map(str, ids[k:k + 150]))}")
        for p in js.get("people", []):
            rows.append({"player": int(p["id"]), "bat_side": p.get("batSide", {}).get("code"),
                         "pitch_hand": p.get("pitchHand", {}).get("code")})
        time.sleep(0.2)
    pd.DataFrame(rows).to_parquet(out / "hands.parquet", index=False)
    print("hands", len(rows), "of", len(ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["alignment", "hands"])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.what == "alignment":
        alignment(a.states, a.out, a.seasons)
    else:
        hands(a.states, a.out)


if __name__ == "__main__":
    main()
