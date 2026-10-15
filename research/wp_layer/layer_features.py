"""
Features for a correction layer on top of the fitted engine (wp_engine.py).

Everything is causal: a row's features use only games played on earlier
dates (team strength, starter quality) and the state of its own game at
the moment of the plate appearance.

Built per plate appearance:
  p_eng   engine home WP before the PA
  L       logit(p_eng)
  r       share of regulation outs still to be played, floored at one half
  d_prev  home minus away run differential per game, previous season
  d_cur   home minus away run differential per game so far this season
          (games on earlier dates), each team shrunk by n / (n + K_TEAM)
  sp      away starter minus home starter FIP-like rate per batter faced
          (previous season + this season on earlier dates, shrunk to the
          league rate with K_SP batters faced); positive favours home
  diff    home minus away score before the PA, clipped to +-10
  re      runs so far (both teams) minus the league mean per half x halves
          completed (the mean comes from the training seasons only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
from wp_engine import Engine, _occ  # noqa: E402

K_TEAM = 30          # games
K_SP = 200           # batters faced
FIP_W = {"home_run": 13.0, "walk": 3.0, "intent_walk": 3.0, "hit_by_pitch": 3.0,
         "strikeout": -2.0, "strikeout_double_play": -2.0}


def load_pa(states: Path, seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(states / f"pa_{s}.parquet") for s in seasons],
                     ignore_index=True)


def game_table(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per game: date, teams, final runs, starters. Gate: winner matches."""
    pa = pa.sort_values(["game_pk", "at_bat_index"])
    post_h = pa.home_score + np.where(pa.half == "bottom", pa.runs_on_play, 0)
    post_a = pa.away_score + np.where(pa.half == "top", pa.runs_on_play, 0)
    pa = pa.assign(post_h=post_h, post_a=post_a)
    last = pa.groupby("game_pk").tail(1).set_index("game_pk")
    g = last[["date", "season", "home_team", "away_team", "home_won"]].copy()
    g["rs_home"] = last.post_h
    g["rs_away"] = last.post_a
    bad = ((g.rs_home > g.rs_away).astype(int) != g.home_won).sum()
    assert bad == 0, f"{bad} games whose final score does not match home_won"
    first_top = pa[pa.half == "top"].groupby("game_pk").head(1).set_index("game_pk")
    first_bot = pa[pa.half == "bottom"].groupby("game_pk").head(1).set_index("game_pk")
    g["sp_home"] = first_top.pitcher.reindex(g.index)      # home pitches the top
    g["sp_away"] = first_bot.pitcher.reindex(g.index)
    return g


def team_strength(games: pd.DataFrame) -> pd.DataFrame:
    """Per game: d_prev and d_cur (home minus away), from games on earlier dates."""
    long = pd.concat([
        pd.DataFrame({"game_pk": games.index, "date": games.date, "season": games.season,
                      "team": games.home_team, "rd": games.rs_home - games.rs_away}),
        pd.DataFrame({"game_pk": games.index, "date": games.date, "season": games.season,
                      "team": games.away_team, "rd": games.rs_away - games.rs_home}),
    ], ignore_index=True)
    season_rd = long.groupby(["season", "team"]).rd.mean()
    # to date: sum and count over earlier dates (a doubleheader's first game does
    # not count for the second)
    daily = long.groupby(["season", "team", "date"]).rd.agg(["sum", "count"]).reset_index()
    daily = daily.sort_values(["season", "team", "date"])
    grp = daily.groupby(["season", "team"])
    daily["cum_sum"] = grp["sum"].cumsum() - daily["sum"]
    daily["cum_n"] = grp["count"].cumsum() - daily["count"]
    long = long.merge(daily[["season", "team", "date", "cum_sum", "cum_n"]],
                      on=["season", "team", "date"], how="left")
    n = long.cum_n.to_numpy(float)
    per = np.where(n > 0, long.cum_sum / np.maximum(n, 1), 0.0)
    long["cur"] = per * n / (n + K_TEAM)
    prev = season_rd.rename("prev").reset_index()
    prev["season"] += 1
    long = long.merge(prev, on=["season", "team"], how="left")
    long["has_prev"] = long.prev.notna()
    long["prev"] = long.prev.fillna(0.0)
    h = long.merge(games[["home_team"]], left_on="game_pk", right_index=True)
    is_home = h.team == h.home_team
    hh = h[is_home].set_index("game_pk")
    aa = h[~is_home].set_index("game_pk")
    out = pd.DataFrame(index=games.index)
    out["d_prev"] = hh.prev - aa.prev
    out["d_cur"] = hh.cur - aa.cur
    out["has_prev"] = hh.has_prev & aa.has_prev
    return out


def starter_quality(pa: pd.DataFrame, games: pd.DataFrame, league_exclude=()) -> pd.Series:
    """Per game: away SP rate minus home SP rate (positive favours home).
    The league rate the starters are shrunk to leaves out league_exclude (the
    test season)."""
    done = pa[pa.pa_complete]
    w = done.event_type.map(FIP_W).fillna(0.0)
    daily = (done.assign(w=w).groupby(["pitcher", "season", "date"])
             .agg(num=("w", "sum"), bf=("w", "size")).reset_index()
             .sort_values(["pitcher", "season", "date"]))
    keep = ~done.season.isin(list(league_exclude)).to_numpy()
    league = float(w[keep].sum() / keep.sum())
    grp = daily.groupby(["pitcher", "season"])
    daily["c_num"] = grp.num.cumsum() - daily.num
    daily["c_bf"] = grp.bf.cumsum() - daily.bf
    seas = daily.groupby(["pitcher", "season"]).agg(s_num=("num", "sum"), s_bf=("bf", "sum"))
    seas = seas.reset_index()
    seas["season"] += 1
    daily = daily.merge(seas, on=["pitcher", "season"], how="left").fillna({"s_num": 0, "s_bf": 0})

    def rate(pid, season, date):
        key = pd.DataFrame({"pitcher": pid, "season": season, "date": date})
        m = key.merge(daily, on=["pitcher", "season", "date"], how="left")
        num = m.c_num.fillna(0) + m.s_num.fillna(0)
        bf = m.c_bf.fillna(0) + m.s_bf.fillna(0)
        # a starter who has not pitched earlier this season has no daily row on
        # this date: fall back to the previous season alone
        miss = m.c_num.isna()
        if miss.any():
            prev = key[miss.values].merge(seas, on=["pitcher", "season"], how="left")
            num[miss.values] = prev.s_num.fillna(0).to_numpy()
            bf[miss.values] = prev.s_bf.fillna(0).to_numpy()
        return ((num + K_SP * league) / (bf + K_SP)).to_numpy()

    # the starter pitched on the game date, so his own daily row exists and the
    # cumulative columns already exclude that date
    r_home = rate(games.sp_home.to_numpy(), games.season.to_numpy(), games.date.to_numpy())
    r_away = rate(games.sp_away.to_numpy(), games.season.to_numpy(), games.date.to_numpy())
    # FIP-like rate: higher is worse for the pitcher, so away minus home favours home
    return pd.Series(r_away - r_home, index=games.index, name="sp")


def engine_wp(eng: Engine, pa: pd.DataFrame) -> np.ndarray:
    keys = list(zip(pa.inning, pa.half, pa.outs, pa.on_1b.notna(), pa.on_2b.notna(),
                    pa.on_3b.notna(), np.clip(pa.home_score - pa.away_score, -15, 15),
                    pa.scheduled_innings))
    cache: dict = {}
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        v = cache.get(k)
        if v is None:
            inn, half, o, b1, b2, b3, d, s = k
            v = eng.wp(int(inn), half, int(o), (int(b1), int(b2), int(b3)), int(d), int(s))
            cache[k] = v
        out[i] = v
    return out


def build(states: Path, seasons, eng: Engine, league_exclude=()) -> pd.DataFrame:
    """seasons: the seasons whose rows are returned; the previous season is loaded
    for d_prev and the starters' previous-season rates. league_exclude: seasons
    left out of the league rate in sp."""
    need = sorted(set(seasons) | {s - 1 for s in seasons})
    have = [s for s in need if (states / f"pa_{s}.parquet").exists()]
    pa = load_pa(states, have)
    games = game_table(pa)
    ts = team_strength(games)
    sp = starter_quality(pa, games, league_exclude)
    rows = pa[pa.season.isin(seasons)].copy()
    rows["p_eng"] = engine_wp(eng, rows)
    g = ts.join(sp)
    rows = rows.join(g, on="game_pk")
    total = 6 * rows.scheduled_innings
    done = ((rows.inning - 1) * 2 + (rows.half == "bottom")) * 3 + rows.outs
    rows["r"] = np.maximum(1 - done / total, 3 / total)
    rows["halves_done"] = (rows.inning - 1) * 2 + (rows.half == "bottom")
    rows["diff"] = np.clip(rows.home_score - rows.away_score, -10, 10)
    rows["runs_so_far"] = rows.home_score + rows.away_score
    return rows
