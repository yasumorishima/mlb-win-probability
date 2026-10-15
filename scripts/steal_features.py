"""
Stolen-base attempt table with features known before the attempt.

One row per steal attempt of second or third base that happened on a
pitch (row_kind == "outcome", event_type stolen_base_2b/3b or
caught_stealing_2b/3b). Steals of home and pickoff outcomes are left out.

Features use only information available before the game of the attempt:
  * the situation (base, outs, count, disengagements, inning, score),
  * the pitcher's throwing hand,
  * the runner's sprint speed and the catcher's pop time / arm from the
    PREVIOUS season (Savant season tables cover the whole season, so the
    current season's values would include the games being predicted),
  * running counts of attempts and successes for the runner, the catcher
    and the pitcher, over all earlier dates (strictly before the game date),
    shrunk toward the league rate.

Usage:
  python scripts/steal_features.py --states data/states --hf data/hf --out data/steal_attempts.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTEMPT_TYPES = {"stolen_base_2b", "stolen_base_3b", "caught_stealing_2b", "caught_stealing_3b"}
SHRINK = {"runner": 10.0, "catcher": 30.0, "pitcher": 30.0}


def load_attempts(states: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(states.glob("run_*.parquet"))]
    run = pd.concat(frames, ignore_index=True)
    a = run[(run.row_kind == "outcome") & run.event_type.isin(ATTEMPT_TYPES) & run.on_pitch].copy()
    a["success"] = a.event_type.str.startswith("stolen_base").astype(int)
    a["to_third"] = a.event_type.str.endswith("3b").astype(int)
    a["date"] = pd.to_datetime(a["date"])
    batting_home = a.half == "bottom"
    a["bat_diff"] = np.where(batting_home, a.home_score - a.away_score, a.away_score - a.home_score)
    return a.sort_values(["date", "game_pk", "at_bat_index", "event_index"]).reset_index(drop=True)


def _prior_counts(a: pd.DataFrame, who: str) -> pd.DataFrame:
    """Attempts / successes by `who` on dates strictly before each row's date."""
    daily = (a.groupby([who, "date"]).success.agg(["size", "sum"]).rename(
        columns={"size": "att", "sum": "succ"}).reset_index().sort_values([who, "date"]))
    daily["att_before"] = daily.groupby(who).att.cumsum() - daily.att
    daily["succ_before"] = daily.groupby(who).succ.cumsum() - daily.succ
    return daily[[who, "date", "att_before", "succ_before"]]


def add_features(a: pd.DataFrame, hf: Path) -> pd.DataFrame:
    a = a.copy()
    a["season"] = a.date.dt.year
    # league rate before each date (expanding, strictly earlier dates)
    by_day = a.groupby("date").success.agg(["size", "sum"]).sort_index()
    cum_att = by_day["size"].cumsum() - by_day["size"]
    cum_succ = by_day["sum"].cumsum() - by_day["sum"]
    league = (cum_succ / cum_att).fillna(0.78)
    a["league_rate_before"] = a.date.map(league)

    for who in ("runner", "catcher", "pitcher"):
        pc = _prior_counts(a, who)
        a = a.merge(pc, on=[who, "date"], how="left")
        k = SHRINK[who]
        a[f"{who}_att_before"] = a.pop("att_before").fillna(0)
        succ = a.pop("succ_before").fillna(0)
        if who == "runner":
            # runner's own success rate, shrunk to the league rate
            a["runner_rate_shrunk"] = (succ + k * a.league_rate_before) / (a.runner_att_before + k)
        else:
            # rate the runners achieved against this catcher / pitcher
            a[f"{who}_allowed_rate_shrunk"] = (succ + k * a.league_rate_before) / (a[f"{who}_att_before"] + k)

    sprint = pd.read_parquet(hf / "sprint_speed.parquet")[["player_id", "season", "sprint_speed"]]
    sprint = sprint.groupby(["player_id", "season"], as_index=False).sprint_speed.mean()
    sprint["season"] += 1                      # previous season's value
    a = a.merge(sprint.rename(columns={"player_id": "runner", "sprint_speed": "runner_sprint_prev"}),
                on=["runner", "season"], how="left")

    cat = pd.read_parquet(hf / "catcher.parquet")
    cat = cat[["player_id", "season", "pop_2b_sba", "maxeff_arm_2b_3b_sba"]].groupby(
        ["player_id", "season"], as_index=False).mean()
    cat["season"] += 1
    a = a.merge(cat.rename(columns={"player_id": "catcher", "pop_2b_sba": "catcher_pop_prev",
                                    "maxeff_arm_2b_3b_sba": "catcher_arm_prev"}),
                on=["catcher", "season"], how="left")
    return a


FEATURES = [
    "to_third", "outs", "balls", "strikes", "diseng", "late", "close", "lefty",
    "runner_sprint_prev", "runner_sprint_missing",
    "runner_rate_shrunk", "runner_att_log",
    "catcher_pop_prev", "catcher_pop_missing", "catcher_arm_prev",
    "catcher_allowed_rate_shrunk", "pitcher_allowed_rate_shrunk",
]


def design(a: pd.DataFrame, fill: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Model matrix. `fill` (means for missing values) comes from training rows."""
    X = pd.DataFrame(index=a.index)
    X["to_third"] = a.to_third
    X["outs"] = a.outs_before
    X["balls"] = a.balls
    X["strikes"] = a.strikes
    X["diseng"] = a.disengagements_before.clip(upper=2)
    X["late"] = (a.inning >= 7).astype(int)
    X["close"] = (a.bat_diff.abs() <= 1).astype(int)
    X["lefty"] = (a.pitch_hand == "L").astype(int)
    X["runner_sprint_missing"] = a.runner_sprint_prev.isna().astype(int)
    X["catcher_pop_missing"] = a.catcher_pop_prev.isna().astype(int)
    X["runner_rate_shrunk"] = a.runner_rate_shrunk
    X["runner_att_log"] = np.log1p(a.runner_att_before)
    X["catcher_allowed_rate_shrunk"] = a.catcher_allowed_rate_shrunk
    X["pitcher_allowed_rate_shrunk"] = a.pitcher_allowed_rate_shrunk
    for c in ("runner_sprint_prev", "catcher_pop_prev", "catcher_arm_prev"):
        X[c] = a[c]
    if fill is None:
        fill = {c: float(X[c].mean()) for c in ("runner_sprint_prev", "catcher_pop_prev", "catcher_arm_prev")}
    for c, v in fill.items():
        X[c] = X[c].fillna(v)
    return X[FEATURES], fill


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    att = add_features(load_attempts(a.states), a.hf)
    att.to_parquet(a.out, index=False)
    print(f"attempts {len(att)}  success rate {att.success.mean():.3f}")
    print(att.groupby("season").success.agg(["size", "mean"]))


if __name__ == "__main__":
    main()
