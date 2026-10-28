# Pre-registration: PENR_RECAL (bullpen roles refined: five lead states, weak pooling) on 2026

Status: FROZEN (2026-10-28, cloud). Nothing below changes after the commit that set this line, except
in a dated AMENDMENTS section written before the score step is run. Independent review before the
freeze (read-only subagent, no 2026 outcome read): blocking points fixed — the comparison pin pointed at
this test's own output instead of `results_penw_2026.json`; gates and frozen list filled; wording on
selection (edge of the grid, optimistic estimate, KAPPA 0 attempt) and on the reading added.

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
- `size_pen_x2.py --window 60` (no new option) vs frozen `size_pen_x.py --window 60`, 2022, 40 games:
  max difference 0; both equal the sizing file `penB_w60_2022` on those rows (max difference 0).
- 2026 path: `size_pen_x2.py --season 2026 --window 60 --lead5 --kappa 0.25 --load-seasons 2021,...,2026
  --games 40` vs the frozen parts: 3,092 rows, max difference 0.
- `dry` on 2025: PENR_RECAL 0.152295, PENW_RECAL 0.152524, −0.000229 (95% −0.000301 to −0.000159).
- `predict` written to `published/penr_2026.parquet` (184,327 rows, no outcome read).

## Frozen files (md5)
8a09df835f8911847e24cc67668d9042  research/player_layer/test_penr_2026.py
9cdb52ed7313905279ffaa2cec4e48be  research/player_layer/size_pen_x2.py
9873dd9c9e0144a71d7c676c1b9940b8  research/player_layer/size_pen_x.py
0d61f77c584cc7ef414c9c198aa9e7ec  research/player_layer/test_pen_2026.py
45a794a88aded87976ec4f5567f2be1d  research/player_layer/test_penw_2026.py
a71518eba9ad596861857508619bc270  research/player_layer/size_stack.py
43ddb22adead3c254fb8d488267bd374  research/player_layer/starter_pc.py
6883dd4c9f13b64f0073e06923595312  research/player_layer/game.py
49a5462584d4a3b546f14dd01d996c47  research/player_layer/game_pen.py
24cd633fdb951a6a5a195cf7f17ec149  research/player_layer/layer.py
6ec9535345c1098f069babaf15160269  research/player_layer/pa_rates.py
033bb61b84531f594545611cc45d24a9  research/player_layer/size_context.py
d9a6d77695ed33e4f0bf41ec9d945ccf  research/player_layer/size_game.py
1a5097807220a8b0186ebe85252694ca  research/player_layer/size_game2.py
99446952547af94e88220d61a0acf272  research/player_layer/size_hand.py
14e0b9a891fb36a6b7e6ed61f194ea84  research/player_layer/rates_v2.py
4681783920315619f4a4b58abffe712d  research/player_layer/size_pa.py
ca649a877f4e83d894970372fd2c596a  research/player_layer/size_runs.py
5c3c820b39d36e36191ef3b68c15f4f3  research/player_layer/size_vs_base.py
b979748de5cd4e160981485454a08dd7  research/wp_layer/layer_features.py
12972ac8bb23a2dc8c152c07cbfd7a56  wp_engine.py
4ef5f25b9a2efcbc8917799aaa6cef9c  research/player_layer/published/penr_recal_coef.json
4d218e568a0deb7af0272192eb847815  research/player_layer/published/penw_recal_coef.json
8257406c8c65d2e2c4349a6debf6fdf5  research/player_layer/published/size_vs_pen_w60.json
b67166d2ce6bb8903c33c3aaae5e1e84  research/player_layer/published/size_vs_penw_l5.json
0a9e8af77881990eab6e9d864a86ef80  research/player_layer/published/size_vs_penw_l6.json
0f5b3b8c18a6deb96f3e5136b994e648  research/player_layer/published/size_vs_penw_l5k15.json
a49f82ae0afc005d159936021d02ecf1  research/player_layer/published/size_vs_penw_l5k5.json
21d4b9bc2f45f0315a61d388337196b6  research/player_layer/published/size_vs_penw_l5k1.json
a63fefa4b53107ebcb1a537f8e907d5a  research/player_layer/published/size_vs_penw_l5k025.json
07f7dec279c9c10641fa519d1f8f8a94  research/player_layer/published/penB_w60_l5_k0_2026_part0.parquet
92519799480d6ff01f777c480d4c08a6  research/player_layer/published/penB_w60_l5_k0_2026_part1.parquet
2bd4325ddbb3970f0d73fad6292d0d0e  research/player_layer/published/penB_w60_l5_k0_2026_part2.parquet
631d5db118d5af6fb42fb4dde5d89929  research/player_layer/published/penB_w60_l5_k0_2026_part3.parquet
fab2b175b61f5aefa13f324c3923a3e7  research/player_layer/published/penr_2026.parquet
44a1af9d14fa053e8db177a0f048c0ac  research/player_layer/published/penw_2026.parquet
6e192cf1c4f387948424b9e3346bbd20  research/player_layer/published/results_penw_2026.json
c9e35870ab157190d2d81b1f26b23b87  research/wp_layer/published/pred_2026.parquet

## AMENDMENTS
(none)
