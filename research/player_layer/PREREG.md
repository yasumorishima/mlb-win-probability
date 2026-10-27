# Pre-registration: does the player layer predict 2026 better than the engine and the team layer?

Status: DRAFT (FROZEN in the commit that changes this line; nothing below
changes after that commit except in a dated AMENDMENTS section written
before the score step is run).

## Why
The engine treats every player as league average. The player layer
(`README.md` in this directory) gives every remaining half-inning a run
distribution for the actual lineup, the pitcher expected in that inning,
the defense and the park, and turns it into home win probability with the
engine's rules. It is the base for comparing bench options (pinch hitter,
pitching change, bunt). This test asks whether it predicts the 2026
regular season better than the engine and than the team layer now in the
app.

## What is already known (not blind)
- Sizing on 2022-2025 only, each season with tables fitted on the other
  four (`README.md`, `published/size_*.json`): whole-game layer (v1) WP
  Brier −0.00203 vs the league engine (95% −0.00277 to −0.00127), every
  season negative (2022 −0.0030, 2023 −0.0018, 2024 −0.0023, 2025
  −0.0011); P(≥1 more run this half) log loss −0.0019 with the current-half
  layer. Parameters (shrinkage, park strength, starter through the 6th)
  were chosen on these seasons. The second round (handedness, starter
  length, fielders' OAA) did not change the WP Brier, so v1 is the arm.
- 2026 has been used before in this repository: the engine's 2026
  calibration was read (too high in the 0.3-0.5 band, 3.3 points), the team
  layer was scored on 2026 (TEAM − RECAL −0.00151, TEAM supported), and the
  steal study used 2026 attempts. That motivated part of this work. No
  player-layer prediction on 2026 has been computed or read before this
  file is frozen.
- The team layer's sizing Brier (TEAM 0.15513 vs ENGINE 0.15718, mean over
  2022-2025 held-out seasons) is of the same size as the player layer's
  (−0.00203 vs league). H2 is therefore not expected to be decided either
  way with confidence.
- The comparison file `research/wp_layer/published/pred_2026.parquet`
  holds the outcome `y` next to the ENGINE and TEAM predictions. Before the
  score step only its schema, row count and md5 were read.

## Data and model (`test_2026.py`, frozen code)
- `states-v1` tables, md5s as in `research/wp_layer/run_2026.py`
  (checked by the script). Park factors: HF `yasumorishima/mlb-stats`
  revision `36e2c3f7bdcbcf400bad7ce9c49f52ce7ab36b0b`, `park_factors.parquet`
  md5 `3f1c06008ec1fec5012a8f50338b9687`; 2026 games use the 2025 rows
  (3-year factors ending 2025); Athletics and Rays home games use average
  factors (their home park changed inside that window).
- Engine: `engine_tables.json` (fitted on 2021-2025). Class tables and
  league class rates: 2021-2025. Batter and pitcher records: every
  completed plate appearance of 2021-2026 on dates before the plate
  appearance. Defense and bullpen: the 2026 season on earlier dates.
  Lineups as known at the moment (last occupant of each slot; the lineup
  card for slots that have not batted). Parameters fixed in `layer.PARAMS`
  and `game.STARTER_LAST` (= 6), park strength 0.5.
- Rows: every 2026 regular-season plate appearance of the tables (184,327,
  the rows of `pred_2026.parquet`).

## Arms
- PLAYER: the whole-game player layer v1.
- ENGINE: the engine as it is (recomputed by the script; must equal
  `p_ENGINE` of `pred_2026.parquet` to 1e-9 or the run stops).
- TEAM: `p_TEAM` of `pred_2026.parquet` (the frozen team layer).

## Hypotheses (one-sided, alpha 0.05, Holm over the three)
- H1: Brier(PLAYER) < Brier(ENGINE) for home win.
- H2: Brier(PLAYER) < Brier(TEAM) for home win.
- H3: log loss of P(≥1 more run in the rest of this half-inning),
  PLAYER < ENGINE, on the rows whose half can be played out (not the
  bottom of the last scheduled inning or later).
Bootstrap over games (2,000 resamples, seed 20261028; a game's rows move
together; arms resampled together); one-sided p = (1 + #{resampled
difference ≥ 0}) / 2001. A hypothesis is SUPPORTED only if Holm passes it.
95% percentile intervals for every difference. Resolution: on one season
the sizing intervals suggest a 95% half-width near 0.0015 for H1, so an
effect like 2025's (−0.0011) may not be detected; a NOT SUPPORTED result
is not evidence that players do not matter.

## Reported, not tested
Log loss, calibration in tenths, the 0.3-0.5 band gap of each arm, Brier
by game stage (innings 1-3, 4-6, 7+), the run-level log losses. The dry
run on 2025 (`--dry`, fitted on 2021-2024, no TEAM arm) is reported with
its numbers as a check of the pipeline, not as evidence.

## What follows
- H1 supported → the player layer becomes the base of the bench screen's
  numbers (with the disclosure that 2026 had been used before); the app
  keeps the frozen engine / team layer for the headline WP until the
  2027 prospective check.
- H2 decides which layer is shown as the headline WP after 2027
  confirmation; with H2 not supported, both stay and are shown with their
  2026 numbers.
- 2027: the same frozen code is scored prospectively on the 2027 regular
  season (rule to be registered before opening day).

## Frozen files (md5)
(filled in the freezing commit)

## AMENDMENTS
(none)
