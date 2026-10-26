# Pre-registration: does Claude estimate bunt success and late-inning on-base chances better than a model?

Status: DRAFT (becomes FROZEN in the commit that changes this line; nothing
below changes after that commit except in a dated AMENDMENTS section written
before the affected result is read).

## Why this question
The app's engine values a sacrifice bunt and an intentional walk with
league-average players. The two player-specific numbers behind those calls
are (1) how likely this batter's bunt is to move the lead runner up and
(2) how likely this batter is to reach base in a late, first-base-open
situation (the intentional-walk question is this batter against the next).
Both can be checked against what happened. If Claude cannot estimate them
better than a plain statistical model, its bunt / walk advice adds nothing
beyond the model on these numbers. The design copies the steal study
(`research/steal_claude/PREREG.md`, H1/H2 not supported).

## Data
- `pa_<season>.parquet`, release `states-v1`, 2021-2026, md5s equal to
  `FROZEN_MD5` in `research/wp_layer/run_2026.py` (checked 2026-10-26 in
  cloud). Regular season only (built from `gameType=R`).
- Part BUNT (`bunt_units`): completed plate appearances whose result
  description contains "bunt" (sacrifice, bunt hit, bunt out, strikeout on
  a foul bunt), 0 or 1 out, runner on first and/or second, third base empty.
  y = 1 when the lead runner (on second if there is one, else on first)
  ends the play on a later base or scores, read from base occupancy after
  the play with the batter excluded (a pinch runner after the play carries
  a new id; checked on 2024-2025: every `sac_bunt` is 1, every strikeout and
  double play is 0). Bunts that were fouled off and followed by a swing are
  not in the data (only the PA-ending bunt is visible).
- Part OBP (`obp_units`): completed plate appearances, inning >= 7, first
  base empty, a runner on second and/or third, not an intentional walk or
  catcher's interference. y = 1 for single, double, triple, home run, walk,
  hit by pitch.
- Records (strictly before the date of the unit, since 2021): batter bunts
  in the BUNT population and their successes; batter plate appearances and
  times on base (all completed PAs except IBB / catcher's interference);
  pitcher batters faced and times on base allowed (same definition).
- League rate: this season's rate on earlier dates (BUNT: in the BUNT
  population; OBP: of all completed PAs as above); the previous season's
  full rate until the season has 100 bunts / 2,000 PAs.
- Player and team names from the MLB Stats API (`names.json`).

## Periods and counts (counted before freezing; no outcome of the test or control rows was looked at)
| | BUNT | OBP |
|---|---|---|
| Training for A: 2023-03-30 to 2026-06-30 | 2,655 (incl. control and pilot) | 23,743 (incl. control and pilot) |
| Test: 2026-07-01 to the end of the regular season | 498 in 372 games | 3,204 in 927 games |
| Memory control: random, 2026-04-01 to 06-30 (seed 20261026) | 300 | 300 |
| Pilot: random June 2026, outside the control (seed 20261026) | 20 | 20 |

Training rows were used to check labels and that A fits (training log loss
A 0.4791 vs league rate 0.4859 for BUNT; 0.6297 vs 0.6326 for OBP). Test
and control rows were not subsampled and their outcomes were not read.

## Arms (each part separately)
- A (model): L2 logistic regression (C = 1.0, standardized features,
  intercept unpenalized) with logit of the league rate as an offset
  (coefficient fixed at 1), fitted once on the training period. Features:
  BUNT - batter bunt record shrunk toward the league rate (k = 5), log(1 +
  earlier bunts), log(1 + earlier PAs), batter on-base record shrunk (k = 200
  toward .310), lead runner on second, two runners, one out, inning >= 7 and
  |score difference| <= 1. OBP - batter and pitcher on-base records shrunk
  toward the league rate (k = 200 each), log(1 + PAs), log(1 + batters
  faced), runner on second, runner on third, outs, extra innings, score
  difference clipped to +-4, |difference| <= 1. For the control rows A is
  refitted on 2023-03-30 to 2026-03-31 (A_ctrl) so that it has not seen them.
- B (Claude): `claude-opus-5-5`, prompt `SYSTEM[part]` in `prompts.py`;
  input = date, population and success definition, league rate, situation,
  team names, batter / pitcher names with the raw records above (see
  `sample_payload.json`). Claude may use what it knows about the players up
  to the day before the date and nothing about the game or later.
- C (Claude + model): as B plus A's probability (A_ctrl on control rows),
  described as a logistic regression on the listed facts.
- Reference only: the league rate (L).

Calls go through the Message Batches API without server-side fallbacks; the
answering model and token usage are recorded for every request, and every
response is written to the raw log before it is parsed. Request ids carry
the arm ("B-..." / "C-..."); collection and analysis refuse a record of the
wrong arm. A request that errors, is refused, or whose output is not valid
JSON with 0 <= p_success <= 1 is resubmitted once; if it fails again it is
scored with A's probability, and the count is reported per part and split
(stated in the summary if above 2%).

`selftest.py` (synthetic data with known answers, the causal records
recomputed by brute force, the bunt labels by event type) passed before
freezing, and each of its checks was shown to fail on a planted defect
(records counting the unit's own date, a wrong bunt label, failures filled
with 0.5 instead of A, A instead of A_ctrl on control rows, a record of the
wrong arm accepted).

## Effort level and cost cap (fixed rule, applied after the pilot)
The pilot makes 40 live calls: per part, 10 at effort "low" and 10 at
"medium", B and C alternating. From the mean cost per call, the batch cost
of the full run (B and C on test and control of both parts = 8,604 calls,
plus 100 repeat calls) is projected at half the live price. The run uses
"medium" if that projection is at most USD 40, otherwise "low". The prompt
may change after the pilot only to fix parsing; any change is recorded
under AMENDMENTS. `run_claude.py` refuses any batch that would bring the
projected total above USD 45. The steal study's medium calls averaged about
1,160 input and 170 output tokens (about USD 0.004 per batch call), so the
expected total is about USD 35.

## Hypotheses (one-sided, alpha = 0.05, Holm over the four)
- H1_bunt: mean log loss of C < that of A on the BUNT test set.
- H2_bunt: mean log loss of B < that of A on the BUNT test set.
- H1_obp: mean log loss of C < that of A on the OBP test set.
- H2_obp: mean log loss of B < that of A on the OBP test set.
Paired bootstrap over games (10,000 resamples, seed 20261026) of the
difference in mean log loss; p = share of resamples with difference >= 0.
Probabilities are clipped to [0.01, 0.99] for every arm. Differences are
reported with 95% intervals. By the steal study's scale (95% half-width
0.0065 at 1,843 attempts), the half-width is expected near 0.012 for BUNT
(498) and 0.005 for OBP (3,204); a null is read as "no difference of the
size the interval excludes", not as "no difference".

## Memory check and how it changes the reading (rule fixed in advance)
Claude's training data reaches mid-2026, so it may know first-half
outcomes, and the test starts at that boundary. Per part and arm,
D = log loss of A minus log loss of the arm, on the control (with A_ctrl)
and on the test set. If the 95% bootstrap interval of D_control - D_test
lies above 0, the arm is reported as "consistent with memorization" (not
proof). August-September units are reported with the B - A and C - A
differences and their 95% intervals. A supported hypothesis is described
as skill only if the arm's memory check does not fire, or if its
August-September interval also lies entirely below 0; otherwise it is
reported as supported on the full test set but not claimed as skill.

## Reported regardless of outcome
Per part: log loss, Brier score, AUC and mean prediction of every arm;
calibration in tenths; units whose batter has little history (BUNT: fewer
than 3 earlier bunts; OBP: fewer than 100 earlier PAs); mean |C - A| and
the correlation between C's adjustment (C - A) and the residual (y - A);
for 100 random test units, B is asked a second time and the spread between
the two answers is reported; tokens and cost. Results are published
whatever they are.

## Disclosure
2026 was already used in this repository: the team layer was scored on
every 2026 plate appearance (win probability, not these outcomes) and the
steal study on 2026 steal attempts. No bunt or late-inning on-base outcome
of the test or control rows has been looked at for this study.

## What this does not claim
Whether bunting or walking was the right decision: bunts are chosen by the
team, and the OBP population excludes plate appearances that ended in an
intentional walk, so outcomes say nothing about the choices not made.

## Frozen files (md5)
(filled in the freezing commit)

## AMENDMENTS
(none)
