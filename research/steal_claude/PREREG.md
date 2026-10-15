# Pre-registration: does Claude improve stolen-base success estimates?

Status: FROZEN. Frozen = the commit that first contains this
file with "Status: FROZEN"; nothing below changes after that commit except
in a dated AMENDMENTS section written before the affected result is read.
The frozen commit also contains the code that builds the inputs
(`build_table.py`), calls the API (`run_claude.py`), scores the result
(`analyze.py`), the prompt (`strategist.py`, `SYSTEM_STEAL`), the feature
code (`scripts/steal_features.py`) and one example input
(`sample_payload.json`).

## Why this question
The app asks Claude what to do before the next pitch. For a steal, the
engine gives the break-even success rate; the decision turns on how likely
this runner is to make it against this pitcher and catcher. That
probability is the part of the advice that can be checked against what
happened. If Claude cannot estimate it better than a plain statistical
model, its steal advice adds nothing beyond the model.

## Data
- Tables built by `scripts/build_states.py` (MLB Stats API, 2021-2026,
  release `states-v1`).
- Attempts: runner outcome rows with event_type stolen_base_2b/3b or
  caught_stealing_2b/3b and on_pitch = true. Steals of home, pickoff
  outcomes and attempts not on a pitch are out. Label: 1 when a stolen base
  is credited; caught stealing, including caught stealing on an error, is 0.
- Features (`scripts/steal_features.py`): base, outs, balls, strikes,
  disengagements before the pitch (capped at 2), inning >= 7, |score
  difference| <= 1, pitcher left-handed, runner sprint speed of the
  previous season, catcher pop time and arm of the previous season (with
  missing flags), runner / catcher / pitcher attempt records on earlier
  dates shrunk toward this season's league rate (k = 10 / 30 / 30),
  log(1 + runner prior attempts). Previous-season Savant values only: the
  current season's tables include the games being predicted.
- League rate: success rate of earlier dates of the same season; the
  previous season's full-year rate until the season has 300 attempts.
  Success rates drift (2023 .843, 2024 .833, 2025 .821, 2026 to June .805).

## Periods and counts (counted before freezing; no outcomes of the test period were looked at)
- Training for A: 2023-03-30 to 2026-06-30 (14,427 attempts, including
  the control and pilot rows).
- Test: 2026-07-01 to the end of the 2026 regular season, all 1,843
  attempts in 863 games. Not subsampled.
- Memory control: 300 random attempts from 2026-04-01 to 2026-06-30
  (seed 20261015).
- Pilot: 40 random June 2026 attempts outside the control (seed 20261015).

## Arms
- A (model): L2 logistic regression on the standardized features, C = 1.0,
  with the logit of the league rate as an offset (coefficient fixed at 1),
  fitted once on the training period. For the control rows A is refitted
  on 2023-03-30 to 2026-03-31 (A_ctrl) so that it has not seen them.
- B (Claude): `claude-opus-5-5`, prompt `SYSTEM_STEAL`; input = date,
  population definition, league attempts / successes this season before
  the date, situation, team names, runner / pitcher / catcher names with
  raw attempt and success counts since 2021 before the date, previous-
  season Savant values with league mean and sd (units in the key names).
  Claude may use what it knows about the players up to the day before the
  date and nothing about the game or later.
- C (Claude + model): as B plus A's probability (A_ctrl on control rows)
  described as a logistic regression on the listed facts.
- Reference only: the league rate (L).

Calls go through the Message Batches API from the Codespace that holds the
key, without server-side fallbacks; the answering model and token usage
are recorded for every request. Request ids carry the arm ("B-..." /
"C-..."), and collection and analysis refuse a record of the wrong arm.
`selftest.py` checks the analysis on synthetic data with known answers
(including swapped arms) and is run before the first call. A request that errors, is refused, or
whose output is not valid JSON with 0 <= p_success <= 1 is resubmitted
once; if it fails again it is scored with A's probability, and the count
is reported per split (stated in the summary if above 2%).

## Effort level (fixed rule, applied after the pilot)
The pilot calls 20 attempts at effort "low" and 20 at "medium" (live API,
B and C alternating). From the mean cost per call, the batch cost of the
full run (B and C on test and control, plus 100 repeat calls) is projected
at half the live price. The run uses "medium" if that projection is at most
USD 40, otherwise "low". The prompt may change after the pilot only to fix
parsing; any change is recorded under AMENDMENTS.

## Hypotheses (one-sided, alpha = 0.05, Holm over the two)
- H1: mean log loss of C < that of A on the test set.
- H2: mean log loss of B < that of A on the test set.
Paired bootstrap over games (10,000 resamples, seed 20261015) of the
difference in mean log loss; p = share of resamples with difference >= 0.
Probabilities are clipped to [0.01, 0.99] for every arm. Differences are
reported with 95% intervals. With 1,843 attempts, differences smaller than
about 0.010 in log loss cannot be detected; a null is read as "no
difference of that size", not as "no difference".

## Secondary (one-sided, alpha = 0.05, outside the Holm family)
- S1: B's gain over A (log loss of A minus B) is larger on test attempts
  where the runner's previous-season sprint speed or the catcher's pop
  time is missing than where both are present (games resampled jointly
  for both groups). AUC of A and B on the missing rows is reported with it.

## Memory check and how it changes the reading (rule fixed in advance)
Claude's training data reaches mid-2026, so it may know first-half
outcomes, and the test starts at that boundary. D = log loss of A minus
log loss of the Claude arm, on the control (with A_ctrl) and on the test
set. If the 95% bootstrap interval of D_control - D_test lies above 0, the
arm is reported as "consistent with memorization" (the check fires on
roughly 5-8% of equal-skill synthetic data, so it is not proof).
August-September attempts are reported with the B - A and C - A
differences and their 95% intervals. A supported H1 or H2 is described as
skill only if the arm's memory check does not fire, or if its
August-September interval also lies entirely below 0; otherwise the
hypothesis is reported as supported on the full test set but not claimed
as skill.

## Reported regardless of outcome
Brier score and AUC of every arm; calibration in tenths; runners with
fewer than 5 earlier attempts; the correlation between C's adjustment
(C - A) and the residual (y - A); for 100 random test attempts, B is asked
a second time and the spread between the two answers is reported; tokens
and cost. Results are published whatever they are.

## What this does not claim
Whether stealing was the right decision: attempts are chosen by the team,
so outcomes say nothing about attempts that were not made.

## AMENDMENTS
(none)
