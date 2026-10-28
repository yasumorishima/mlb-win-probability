# Pre-registration: PENT_RECAL (lead-state smoothing TAU 1 instead of 20) on 2026

Status: DRAFT (not frozen; nothing has been scored).

## Why
PENR_RECAL (60-day usage, five lead states, KAPPA 0.25; `PREREG_PENR.md`, `RESULTS_PENR_2026.md`) is the
look-back view's win probability. Its usage mixture weights a reliever in a lead state by his relief plate
appearances in that state plus TAU (= 20) pseudo plate appearances spread by his share of the inning's
usage, i.e. the state share is smoothed toward the inning share. Sizing after the PENR test, on 2022-2025
only (2026 not read), every arm recalibrated out of fold as PENR_RECAL and compared with it
(`size_vs_base.py --base-tag penB_w60_l5_k0`, `size_pen_x3.py --tau`):

| TAU | minus PENR_RECAL (95%) | by season 2022 / 2023 / 2024 / 2025 | file |
|---|---|---|---|
| 5 | −0.000164 (−0.000180 to −0.000149) | −0.000118 / −0.000174 / −0.000194 / −0.000171 | `size_vs_penr_t5.json` |
| **1** | **−0.000246 (−0.000272 to −0.000221)** | **−0.000172 / −0.000264 / −0.000293 / −0.000255** | `size_vs_penr_t1.json` |
| 0 | −0.000203 (−0.000249 to −0.000156) | −0.000117 / −0.000247 / −0.000277 / −0.000169 | `size_vs_penr_t0.json` |

Values tried, in this order and only after the PENR 2026 result was known: 5, 1, 0 (no others; 0 was run after 1 was seen to be the best so far). Command per value: `size_vs_base.py --base /root/claude-scratch/player/penk025 --base-tag penB_w60_l5_k0 --cand <dir> --cand-tag penB_w60_l5_k0_t<TAU>p0`. TAU 1 was chosen because its 2022-2025 mean was lowest (the same rule as
the window and KAPPA); it is an interior point of the values tried, picked on the same data that fits the
recalibration, so the −0.00025 is an optimistic estimate and the 2025 rehearsal below is not independent.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests (engine, team layer, steal study, player layer v1,
BLEND, bench-call study, PEN_RECAL, PENW_RECAL, PENR_RECAL). The PENR_RECAL result on 2026 is known:
Brier 0.154793, vs PENW_RECAL −0.000184 (−0.000269 to −0.000107), vs TEAM −0.00200
(`published/results_penr_2026.json`); its per-play predictions are published (`penr_2026.parquet`,
`pen_wp_2026.parquet`), the look-back view shows them next to 2026 results, and the game notes
(`research/game_notes/`) were written from them. No 2026 prediction with another TAU has been made or
compared with an outcome.

## Model (frozen)
- 2026 predictions: `size_pen_x3.py --season 2026 --variant B --window 60 --lead5 --kappa 0.25 --tau 1
  --load-seasons 2021,2022,2023,2024,2025,2026 --part K/4` (K = 0..3), stored as
  `published/penB_w60_l5_k0_t1p0_2026_part{K}.parquet`. Everything else as PENR_RECAL's 2026 predictions.
  `size_pen_x3.py` is a copy of the frozen `size_pen_x2.py` plus `--tau` (gates below).
- Recalibration: `test_pent_2026.py fit` on the 2022-2025 out-of-fold sizing predictions (733,260 rows)
  → `published/pent_recal_coef.json`; same terms (L, L·r, L·r², r, r²).
- PENT_RECAL = sigmoid(design · coef) on the 2026 predictions (`predict`); the score step recomputes it from
  the frozen parts and coefficients and stops if it differs.
- Rehearsal on 2025 (`dry`, both layers recalibrated on 2022-2024, no 2026 read): PENT_RECAL 0.152040,
  PENR_RECAL 0.152295 (= the PENR rehearsal), difference −0.000255 (95% −0.000307 to −0.000202).

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(PENT_RECAL) < Brier(PENR_RECAL) (`p_PENR_RECAL` of `published/penr_2026.parquet`, the file
  scored in PREREG_PENR: md5 equal to `pred_md5` in `results_penr_2026.json`, checked by the score step,
  and frozen below).
- H2: Brier(PENT_RECAL) < Brier(TEAM) — a consistency check, not new evidence (PENR_RECAL − TEAM known).
One-sided, game bootstrap (2,000 resamples, seed 20261030, `test_pen_2026.run_tests`),
p = (1 + #{resampled difference ≥ 0}) / 2001, Holm over the two, supported if the adjusted p ≤ 0.05.
Brier on unclipped probabilities. Expected resolution: on 2025 the interval half-width was about 0.00005
and the sizing difference is about −0.00025, so H1 can be supported if the gain carries over; NOT
SUPPORTED is possible even if it is real.
Reported, not tested: log loss and the 0.3-0.5 band gap of PENT_RECAL, PENR_RECAL, TEAM, PENT.

## Run
`test_pent_2026.py predict --out research/player_layer/published` (no outcome read), then
`score --out research/player_layer/published` once. The score step requires Status: FROZEN and checks
every md5 below and the input data md5s (via `test_pen_2026.check_inputs`).
Outputs `published/pent_2026.parquet`, `published/results_pent_2026.json`, `RESULTS_PENT_2026.md`.

## What follows
- H1 supported → the look-back view's win probability series becomes PENT_RECAL (2022-2026 games, 2022-2025
  from the sizing predictions recalibrated with `pent_recal_coef.json`), and the bench options' bullpen
  mixture uses TAU 1 (gate against `size_pen_x3.py` as `check_usage_penr.py` did). The game notes already
  written are not regenerated (no API credit). The usage-pattern pick is unchanged.
- H1 not supported → PENR_RECAL stays; the numbers are published either way.

## Gates (run before the freeze)
(filled before the freeze)

## Frozen files (md5)
(filled at the freeze)

## AMENDMENTS
(none)
