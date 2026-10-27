"""A small synthetic finished game in the shape of the Stats API live feed
(no recorded data is committed): four innings, home wins 1-0.

  top 1-4: the away team goes down in order every inning
  bottom 2: single, stolen base, two outs
  bottom 3: a pinch hitter (home, slot 8) bats first ... in his slot
  bottom 4: a pitching change by the away team, then a home run
"""

import bench_live as BL

SNAP = BL._snapshot()
BAT = sorted(SNAP["_bat"], key=lambda i: -SNAP["_bat"][i].sum())[:24]
SP = [int(k) for k, (st, g) in SNAP["role"].items() if st >= 20 and st == g][:2]
RP = [int(k) for k, (st, g) in SNAP["role"].items() if st == 0 and g >= 40][:4]
HOME_ID, AWAY_ID = 121, 143
HOME_LU, AWAY_LU = BAT[:9], BAT[9:18]
HOME_BENCH, AWAY_BENCH = BAT[18:21], BAT[21:24]
PH = HOME_BENCH[0]
OUT = ("Strikeout", "Groundout", "Strikeout")


def game_feed() -> dict:
    plays = []
    score = {"home": 0, "away": 0}
    slot = {"home": 0, "away": 0}
    lineup = {"home": list(HOME_LU), "away": list(AWAY_LU)}
    pitcher = {"home": SP[0], "away": SP[1]}            # who pitches FOR each team

    def pa(inning, top, outs_before, event, runs=0, post=None, events=None):
        side, fld = ("away", "home") if top else ("home", "away")
        batter = lineup[side][slot[side] % 9]
        slot[side] += 1
        score[side] += runs
        m = {"batter": {"id": batter, "fullName": f"P{batter}"},
             "pitcher": {"id": pitcher[fld], "fullName": f"P{pitcher[fld]}"}, **(post or {})}
        plays.append({"about": {"inning": inning, "isTopInning": top, "isComplete": True,
                                "halfInning": "top" if top else "bottom"},
                      "result": {"type": "atBat", "event": event, "eventType": event.lower().replace(" ", "_"),
                                 "description": f"P{batter}: {event}", "homeScore": score["home"], "awayScore": score["away"]},
                      "matchup": m, "count": {"outs": outs_before + (event in ("Strikeout", "Groundout"))},
                      "playEvents": events or []})

    for inning in range(1, 5):
        for o, ev in enumerate(OUT):                    # top: three up, three down
            pa(inning, True, o, ev)
        if inning == 1:
            for o, ev in enumerate(OUT):
                pa(inning, False, o, ev)
        elif inning == 2:
            r = lineup["home"][slot["home"] % 9]
            pa(2, False, 0, "Single", post={"postOnFirst": {"id": r}})
            pa(2, False, 0, "Strikeout", post={"postOnSecond": {"id": r}},
               events=[{"details": {"eventType": "stolen_base_2b", "description": f"P{r} steals 2nd"}}])
            pa(2, False, 1, "Groundout")
            pa(2, False, 2, "Strikeout")
        elif inning == 3:
            for o, ev in enumerate(OUT):
                evs = []
                if slot["home"] % 9 == 7:               # pinch hitter in slot 8
                    old = lineup["home"][7]
                    lineup["home"][7] = PH
                    evs = [{"details": {"eventType": "offensive_substitution",
                                        "description": f"Pinch-hitter P{PH} replaces P{old}"},
                            "player": {"id": PH}, "battingOrder": "801", "position": {"abbreviation": "PH"}}]
                pa(inning, False, o, ev, events=evs)
        else:
            pitcher["away"] = RP[2]
            pa(4, False, 0, "Home Run", runs=1,
               events=[{"details": {"eventType": "pitching_substitution",
                                    "description": f"Pitching Change: P{RP[2]} replaces P{SP[1]}"},
                        "player": {"id": RP[2]}, "position": {"abbreviation": "P"}}])
            for o, ev in enumerate(OUT):
                pa(4, False, o, ev)

    def roster(lu, bench, pitchers, ph=None):
        d = {}
        for i, b in enumerate(lu):
            d[f"ID{b}"] = {"person": {"id": b, "fullName": f"P{b}"}, "position": {"abbreviation": "RF"}, "battingOrder": f"{i + 1}00"}
        for b in bench:
            d[f"ID{b}"] = {"person": {"id": b, "fullName": f"P{b}"}, "position": {"abbreviation": "RF"}}
        for p in pitchers:
            d[f"ID{p}"] = {"person": {"id": p, "fullName": f"P{p}"}, "position": {"abbreviation": "P"}}
        if ph:
            d[f"ID{ph}"]["battingOrder"] = "801"
        return d

    home = roster(HOME_LU, HOME_BENCH, [SP[0]] + RP[:2], PH)
    away = roster(AWAY_LU, AWAY_BENCH, [SP[1]] + RP[2:])
    people = {k: {"id": v["person"]["id"], "fullName": v["person"]["fullName"]} for r in (home, away) for k, v in r.items()}
    return {
        "gameData": {"game": {"pk": 777, "type": "R", "season": "2026"}, "status": {"detailedState": "Final"},
                     "datetime": {"officialDate": "2026-07-01"},
                     "teams": {"home": {"id": HOME_ID, "name": "Home"}, "away": {"id": AWAY_ID, "name": "Away"}},
                     "players": people},
        "liveData": {"plays": {"allPlays": plays},
                     "linescore": {"scheduledInnings": 9, "teams": {"home": {"runs": score["home"]}, "away": {"runs": score["away"]}}},
                     "boxscore": {"teams": {"home": {"team": {"id": HOME_ID}, "players": home},
                                            "away": {"team": {"id": AWAY_ID}, "players": away}}}},
    }
