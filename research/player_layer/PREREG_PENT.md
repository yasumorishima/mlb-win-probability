# Pre-registration: PENT_RECAL (lead-state smoothing TAU 1 instead of 20) on 2026

Status: FROZEN (2026-10-28, cloud). Nothing below changes after the commit that set this line, except
in a dated AMENDMENTS section written before the score step is run. Independent review before the
freeze (read-only subagent, no 2026 outcome read): no blocking point in the computation; docstring arm
names fixed, selection order stated, gates and frozen list filled. The bootstrap seed is the same as in
PREREG_PENR (20261030), so the resample weights are identical.

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
- `size_pen_x3.py --window 60 --lead5 --kappa 0.25` without `--tau`, and with `--tau 20`, vs frozen
  `size_pen_x2.py` with the same options, 2022, 40 games: max difference 0 (both; same file name).
- 2026 path: `size_pen_x3.py --season 2026 ... --tau 1 --games 40` vs the frozen parts: 3,092 rows, max
  difference 0.
- `dry` on 2025: PENT_RECAL 0.152040, PENR_RECAL 0.152295, −0.000255 (95% −0.000307 to −0.000202).
- `predict` written to `published/pent_2026.parquet` (184,327 rows, no outcome read).

## Frozen files (md5)
5a8aca30c7e23679747cc9f13779de3d  research/player_layer/test_pent_2026.py
2a8a1c3e845a692d93b8517e2aaa3daf  research/player_layer/size_pen_x3.py
9cdb52ed7313905279ffaa2cec4e48be  research/player_layer/size_pen_x2.py
0d61f77c584cc7ef414c9c198aa9e7ec  research/player_layer/test_pen_2026.py
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
72813cf8a5ee766b9ffd5baf2d03e817  research/player_layer/published/pent_recal_coef.json
4ef5f25b9a2efcbc8917799aaa6cef9c  research/player_layer/published/penr_recal_coef.json
e4fc14188ba7a512141d9ed610aecfbc  research/player_layer/published/size_vs_penr_t5.json
816d76a38e4b255b0fd5603291208112  research/player_layer/published/size_vs_penr_t1.json
d47db4528dd5b921b7ffcebd2c78f417  research/player_layer/published/size_vs_penr_t0.json
b316686cca0bb65176358338be5bf06d  research/player_layer/published/penB_w60_l5_k0_t1p0_2026_part0.parquet
2d65986368ce5a85176cafebfd955238  research/player_layer/published/penB_w60_l5_k0_t1p0_2026_part1.parquet
e10bdbadd597b67f0f00fea130392ffc  research/player_layer/published/penB_w60_l5_k0_t1p0_2026_part2.parquet
76cc15102a519b81b9338e8655d5513f  research/player_layer/published/penB_w60_l5_k0_t1p0_2026_part3.parquet
287357f54789b20f2d2e6f4a5e1ebb69  research/player_layer/published/pent_2026.parquet
fab2b175b61f5aefa13f324c3923a3e7  research/player_layer/published/penr_2026.parquet
e4d21a11f53a34bd45cbb552d0903572  research/player_layer/published/results_penr_2026.json
c9e35870ab157190d2d81b1f26b23b87  research/wp_layer/published/pred_2026.parquet

## AMENDMENTS
(none)
