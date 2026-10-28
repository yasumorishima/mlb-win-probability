# Pre-registration: PENR_RECAL (bullpen roles refined: five lead states, weak pooling) on 2026

Status: DRAFT (not frozen; nothing has been scored).

## Why
PENW_RECAL (60-day reliever usage; `PREREG_PENW.md`, `RESULTS_PENW_2026.md`) is the look-back view's
win probability. A placebo on 2022-2025 showed that about half of the usage bullpen's gain comes from
roles (who pitches which inning at which score; README "Where the bullpen gain comes from"). Sizing of
the role estimate after the PENW test, on 2022-2025 only (2026 not read for any of it), every arm
recalibrated out of fold as PENW_RECAL and compared with it (`size_vs_base.py --base-tag penB_w60`):

| arm (all 60-day usage) | minus PENW_RECAL (95%) | by season 2022-2025 | file |
|---|---|---|---|
| five lead states (behind by 3+ / behind by 1-2 / tied / ahead 1-3 / ahead 4+) | −0.0000186 (−0.0000209 to −0.0000162) | all negative | `size_vs_penw_l5.json` |
| six lead states (also ahead by 1 / ahead by 2-3) | +0.0000873 (+0.0000790 to +0.0000966) | all positive | `size_vs_penw_l6.json` |
| five states, pool weight KAPPA 15 (was 30) | −0.0000902 | all negative | `size_vs_penw_l5k15.json` |
| five states, KAPPA 5 | −0.0001471 | all negative | `size_vs_penw_l5k5.json` |
| five states, KAPPA 1 | −0.0001736 | all negative | `size_vs_penw_l5k1.json` |
| **five states, KAPPA 0.25** | **−0.0001789 (−0.0002143 to −0.0001460)** | **all negative** | `size_vs_penw_l5k025.json` |

KAPPA 0 was also attempted (a seventh arm): it divides by zero for a team with no usage in the window,
the runs failed and wrote nothing. KAPPA 0.25 was chosen because its 2022-2025 mean was lowest (as the
60-day window was chosen); it is within 0.000005 of KAPPA 1. It is the smallest value tried (an edge of
the grid) and was picked on the same 2022-2025 data that fits the recalibration, so the −0.00018 is an
optimistic estimate; the 2025 rehearsal below is not independent either (2025 was part of the selection).
File names use `_k0` for KAPPA 0.25 (`int(KAPPA)`); the md5s below identify the files. KAPPA sets the pool weight KAPPA / (KAPPA + usage plate appearances of the available relievers):
with 0.25 the mixture of the team's own relievers is used almost unshrunk whenever any usage exists.
The six-state version being clearly worse than the five-state one is not understood; it is reported,
not explained.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests (engine, team layer, steal study, player layer v1,
BLEND, bench-call study, PEN_RECAL, PENW_RECAL). The PENW_RECAL result on 2026 is known: Brier 0.154977,
vs PEN_RECAL −0.000086 (−0.000133 to −0.000037), vs TEAM −0.00182 (`published/results_penw_2026.json`);
its per-play predictions are published (`penw_2026.parquet`, `pen_wp_2026.parquet`) and the look-back
view shows them next to 2026 results. No 2026 prediction with five lead states or another KAPPA has been
made or compared with an outcome. Selection on 2022-2025: the arms above (six), after the earlier ones
listed in PREREG_PENW; the same seasons fit the recalibration.

## Model (frozen)
- 2026 predictions: `size_pen_x2.py --season 2026 --variant B --window 60 --lead5 --kappa 0.25
  --load-seasons 2021,2022,2023,2024,2025,2026 --part K/4` (K = 0..3), stored as
  `published/penB_w60_l5_k0_2026_part{K}.parquet`. Everything else as PENW_RECAL's 2026 predictions.
  `size_pen_x2.py` is a copy of the frozen `size_pen_x.py` plus `--lead5`, `--lead6`, `--kappa`
  (gates below).
- Recalibration: `test_penr_2026.py fit` on the 2022-2025 out-of-fold sizing predictions (733,260 rows)
  → `published/penr_recal_coef.json`; same terms (L, L·r, L·r², r, r²).
- PENR_RECAL = sigmoid(design · coef) on the 2026 predictions (`predict`); the score step recomputes it
  from the frozen parts and coefficients and stops if it differs.
- Rehearsal on 2025 (`dry`, both layers recalibrated on 2022-2024, no 2026 read): PENR_RECAL 0.152295,
  PENW_RECAL 0.152524 (= `size_vs_pen_w60.json` 2025), difference −0.000229 (95% −0.000301 to −0.000159).

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(PENR_RECAL) < Brier(PENW_RECAL) (`p_PENW_RECAL` of `published/penw_2026.parquet`, the file
  scored in PREREG_PENW: md5 equal to `pred_md5` in `results_penw_2026.json`, checked by the score step,
  and frozen below).
- H2: Brier(PENR_RECAL) < Brier(TEAM) — a consistency check, not new evidence (PENW_RECAL − TEAM known).
One-sided, game bootstrap (2,000 resamples, seed 20261030, `test_pen_2026.run_tests`),
p = (1 + #{resampled difference ≥ 0}) / 2001, Holm over the two, supported if the adjusted p ≤ 0.05.
Brier on unclipped probabilities. Expected resolution: on 2025 the interval half-width was about
0.00007 and the sizing difference is about −0.00018, so H1 can be supported if the gain carries over;
NOT SUPPORTED is possible even if it is real.
Reported, not tested: log loss and the 0.3-0.5 band gap of PENR_RECAL, PENW_RECAL, TEAM, PENR.

## Run
`test_penr_2026.py predict --out research/player_layer/published` (no outcome read), then
`score --out research/player_layer/published` once. The score step requires Status: FROZEN and checks
every md5 below and the input data md5s (via `test_pen_2026.check_inputs`).
Outputs `published/penr_2026.parquet`, `published/results_penr_2026.json`, `RESULTS_PENR_2026.md`.

## What follows
Reading: H1 tests this specific setting (five lead states **and** KAPPA 0.25). Most of the sizing gain
came from KAPPA (five states alone −0.0000186), so a supported H1 does not show that the five lead states
in particular help.
- H1 supported → the look-back view's win probability series becomes PENR_RECAL (2022-2026 games,
  2022-2025 from the sizing predictions recalibrated with `penr_recal_coef.json`). The bench options'
  bullpen mixture and the usage-pattern pick are not changed by this test (they keep the PENW rules /
  the study's 30-day rule); moving the bench mixture to five states and KAPPA 0.25 needs its own gate
  against `size_pen_x2.py`.
- H1 not supported → PENW_RECAL stays; the numbers are published either way.

## Gates (run before the freeze)
(filled before the freeze)

## Frozen files (md5)
(filled at the freeze)

## AMENDMENTS
(none)
