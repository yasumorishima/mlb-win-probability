# A correction layer for the engine: does team strength fix it? (pre-registration)

Written and committed before the evaluation on 2026 is run. The evaluation is
`run_2026.py`, run once after this file is committed. Nothing in the files
listed under "Frozen" changes after that commit.

## Why

The engine (`wp_engine.py`) treats every team as league average. On 2026 its
home win probability is too high in the 0.3–0.5 band (0.351 predicted vs
0.311 observed in 0.3–0.4, 0.447 vs 0.421 in 0.4–0.5; `reports/engine_check_2026.json`).
One likely reason is that team strength carries through a game: a team that
trails is more often the weaker team, and its future scoring is below league
average. This study adds a layer on top of the engine that knows the
teams' strength, and asks whether it predicts the outcome better than the
engine and than a plain recalibration of the engine.

## What is already known (not blind)

- The engine's calibration on 2026, by decile (the table above). That is what
  motivated this study. No layer's prediction on 2026 has been computed and read.
- Sizing on the training seasons only (`size_layer.py` → `size.log`,
  `size_k.py` → `size_k.log`; leave one season out over 2022–2025, 2026 not
  loaded): Brier ENGINE 0.15718, RECAL 0.15703, TEAM 0.15513, SP 0.15647,
  INGAME 0.15703, FULL 0.15505 (mean over the four held-out seasons). The
  engine's gap in the 0.3–0.5 band varies a lot by season (+0.8 to +4.7
  points). TEAM's gain over RECAL is largest early in the game (early third
  −0.0039, middle −0.0012, last −0.0003). In the fold that holds out 2025 the
  strength terms raise the home WP for a stronger home team at every stage
  (the `·r` terms are positive, the `·r²` terms negative, the sum positive
  for r ≤ 1). K_TEAM 10/30/60/120 changes TEAM's Brier by at most 0.00003, so
  the value chosen before the sizing (30) is kept. (An earlier `size_k.py`
  divided the per-stage row counts by two and printed twice these values;
  it was fixed and rerun on 2026-10-16 before freezing.) In the sizing the
  league rate behind `sp` used all of 2021–2025, held-out season included;
  in `run_2026.py` it leaves out the test season.
- **Disclosure: an accidental run of 2026 before this file was written.**
  On 2026-10-15 at 22:47 JST a version of `run_2026.py` without the `--dry`
  switch was started by mistake (a patch that should have added the switch
  failed and the unpatched file was copied and run). It ran to the end and
  wrote `published/results_2026.json`, `published/pred_2026.parquet` and a
  600-byte log. Its output went only to that log file (the terminal showed
  only "started"). I noticed the failed patch at once, stopped to check, did
  not open, print or read any of the three files, and deleted them at 22:51
  JST. The numbers in this file and every choice below come from the
  training seasons only. Choices made after 22:47, all from training-season
  runs: the `--dry` switch (between 22:51 and 22:55); the dry run on 2025 (`dry.log`,
  22:55) showed H3's interval at ±0.026, so H3 was made reported-only and
  Holm was set over H1–H2 (`run_2026.py` edited 22:58); on 2026-10-16, after
  an independent audit, the league rate behind `sp` was made to leave out the
  test season, the frozen inputs are hashed and checked, `size_k.py` was
  fixed, and `size_k.py` and the dry run were rerun (`size_layer.py` and
  `size.log` are unchanged). The times come from the session log; of them,
  only the creation of `dry/` (22:55:26) can be checked against the files.
  That nothing from the accidental run was read cannot be proved from the
  files; it is stated here so the reader can weigh it.

## Data

`states-v1` plate-appearance tables (`scripts/build_states.py`), regular
season. Layers are fitted on 2022–2025 (733,260 plate appearances; 2021
rows are not fitted: they serve as the previous season of 2022 and enter the
league rate behind `sp`) and scored on 2026 (184,327 plate
appearances, the same rows as `reports/engine_check_2026.json`). The engine
tables were fitted on 2021–2025, so on the training rows the engine is
in-sample; on 2026 it is out of sample. Both arms that are compared share this.

## Features (`layer_features.py`, all causal)

For each plate appearance, before it happens:

| name | meaning |
|---|---|
| `L` | logit of the engine's home WP (clipped to 1e-4 … 1−1e-4) |
| `r` | share of the regulation outs still to be played, floored at one half-inning |
| `d_prev` | home minus away run differential per game, previous season |
| `d_cur` | home minus away run differential per game this season, games on earlier dates only, each team shrunk by n/(n+30) |
| `sp` | away starter minus home starter FIP-like rate per batter faced ((13 HR + 3 (BB+IBB+HBP) − 2 K) / BF), previous season plus this season on earlier dates, shrunk to the league rate with 200 BF |
| `diff` | home minus away score, clipped to ±10 |
| `re` | runs so far (both teams) minus the training seasons' mean runs per half-inning × half-innings completed |

## Arms (logistic regression, unpenalised, fitted on 2022–2025)

| arm | terms (plus intercept) |
|---|---|
| ENGINE | the engine as it is |
| RECAL | `L, L·r, L·r², r, r²` |
| **TEAM** | RECAL + `d_prev·r, d_prev·r², d_cur·r, d_cur·r²` |
| SP | RECAL + `sp·r, sp·r²` |
| INGAME | RECAL + `diff·r, re, L·re` |
| FULL | all of the above |

TEAM is the arm under test. RECAL is the control: it has every term of TEAM
except the team-strength ones, so TEAM beating RECAL means the teams'
strength adds information. RECAL is fitted on the training seasons, where the
engine is in-sample and already nearly calibrated (in the dry run RECAL −
ENGINE was −0.00007), so RECAL is close to the identity and cannot learn a
miscalibration that appears only in 2026. TEAM shares that limitation, so H1
compares like with like.

## Hypotheses (2026, one run)

- **H1** Brier(TEAM) < Brier(RECAL).
- **H2** Brier(TEAM) < Brier(ENGINE).
- **H3 (reported with its interval, no verdict)** On the plate appearances
  where the engine says 0.3 ≤ p < 0.5, the absolute gap |mean prediction −
  observed home win rate| is smaller for TEAM than for ENGINE.

Inference: bootstrap over games (2,000 resamples, seed 20261015; a game's
plate appearances move together; both arms are resampled together).
One-sided p = (1 + #{resampled difference ≥ 0}) / 2001. Holm over H1 and H2
at one-sided 0.05. A hypothesis is SUPPORTED only if Holm passes it. 95%
percentile intervals are reported for every difference, H3 included
(difference |gap TEAM| − |gap ENGINE|).

Why H3 has no verdict and the family is H1–H2: the dry run of `run_2026.py`
on a training season (fitted on 2022–2024, scored on 2025; `dry.log`) gave
H3 an interval of −0.026 to +0.026, so one season cannot decide it. That run
also gave the power for H1: interval half-width about 0.0013 (standard error
about 0.00066). The sizing put TEAM − RECAL at −0.0012 to −0.0024 by season
(mean −0.0019). With the first Holm threshold at one-sided 0.025, an effect of
0.0012 would be detected about half the time and 0.0019 about four times in
five. A NOT SUPPORTED result is therefore not evidence that team strength does
nothing. These choices were made from the dry run and the sizing, before any
layer's 2026 number was read.

Reported, not tested: log loss, calibration by decile, the gap with each
arm's own 0.3–0.5 band, Brier by game stage, and these differences with
intervals: RECAL−ENGINE, SP−RECAL, INGAME−RECAL, FULL−TEAM, FULL−RECAL.
No claim is made from a reported-only number.

## What follows from the result

- H1 SUPPORTED → the TEAM layer goes into the app and the API with the
  coefficients of this run (fitted on 2022–2025, not refitted), and `d_prev`
  and `d_cur` defined exactly as here (runs scored minus allowed per game,
  `d_cur` from games on earlier dates, shrunk with K = 30); the app gets the
  runs and games from the MLB Stats API. Before it is shown, the app's
  `d_prev` and `d_cur` are checked against `layer_features.py` on every 2025
  game; any difference above 0.01 runs per game stops the release. The README
  states the 2026 numbers.
- H1 NOT SUPPORTED → no layer in the app, whatever H2 says. The README
  records the result and the engine stays as it is.
- H2 is reported with its verdict; on its own it does not put a layer in the
  app (RECAL is the control that isolates team strength).
- H3 is reported either way; if it is not supported the README keeps saying
  that the 0.3–0.5 band is too high.
- If `run_2026.py` stops before writing its output (a failed check or any
  other error), nothing is read from that run. If the cause is outside the
  frozen files (an input changed, a package missing), the input or
  environment is restored and the run is started again. If a frozen file has
  to change, the change is a separate commit that says what changed and why;
  `FROZEN_MD5`, the row counts, the arms, the seeds and the inference rule are
  never changed. The README lists every restart and its cause.

## Frozen

`PREREG.md`, `run_2026.py`, `layer_features.py`, `size_layer.py` (its
`ARMS` and `design` are imported by the run), `wp_engine.py`,
`engine_tables.json`, and the `states-v1` tables `pa_2021`–`pa_2026`. The run
checks the md5 of the engine, its tables and the six tables against the
values written in `run_2026.py` (`FROZEN_MD5`) and the row counts (733,260 /
184,327), and records the md5 of the three scripts and the inputs in its
output. The gates that can actually fail are these and `has_prev`; the
`notna` checks on `d_prev`, `d_cur` and `sp` and the shared-`game_pk` check
cannot fail by construction (missing values are filled, seasons are split)
and are kept only as documentation.
