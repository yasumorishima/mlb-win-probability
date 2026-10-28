# Pre-registration: PEN_RECAL (player layer + late-inning bullpen by usage, recalibrated) on 2026

Status: FROZEN (2026-10-28, cloud). Nothing below changes after the commit that set this line, except
in a dated AMENDMENTS section written before the score step is run. Independent review before the
freeze (read-only subagent, no 2026 outcome read): blocking points fixed — 2026 predictions pinned
and recomputed at scoring, frozen list filled, disclosures (b)-(d) added, park file md5 checked.

## Why
The headline win probability of the app is the team layer (TEAM, `research/wp_layer`). The player
layer v1 (PLAYER) beat the engine on 2026 but could not be told apart from TEAM
(−0.00003, 95% −0.00098 to +0.00097, `RESULTS_2026.md`), and BLEND did not beat either
(`RESULTS_BLEND_2026.md`). Since then, on 2022-2025 only (2026 not read):
- the late-inning bullpen by usage (`size_pen.py`, variant B: relievers weighted by their share of the
  team's relief plate appearances in that inning and lead state over the previous 30 days, without
  pitchers already used or tired) lowered the Brier by 0.00163 (95% −0.00187 to −0.00141) vs the
  same layer with a pooled bullpen, every season negative (`published/size_pen.json`). That baseline
  is v1 plus the starter-length model of `size_game2.py` ("starter"), so the PEN − PLAYER difference
  tested below includes both changes;
- a logistic recalibration of that layer on out-of-fold predictions (terms L, L·r, L·r², r, r²,
  as `size_stack.py` P_RECAL) removed most of its 0.3-0.5 band bias (+2.68 → +0.55 points) and gave
  Brier 0.153362 vs 0.155185 for the engine with the same team-strength terms as TEAM (E_TEAM):
  −0.0018 overall, −0.00191 / −0.00153 / −0.00208 / −0.00179 in 2022 / 2023 / 2024 / 2025
  (`published/size_stack_pen.json`; the blend weight on the player side was 1.0 in every fold, so
  no blend is tested). The recalibration itself adds little Brier: P_RECAL − P −0.00017
  (95% −0.00035 to +0.00001), and in 2025 it was slightly worse (0.152641 vs 0.152600). It is kept
  in the tested arm because it removes the band bias that a displayed probability should not have.
This test asks whether PEN_RECAL beats TEAM and PLAYER on 2026, to decide the headline number.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests (engine calibration, team layer, steal study,
player layer v1, BLEND, and the bench-call agreement study, which used 2026 manager decisions).
Known before designing this test:
(a) PLAYER ≈ TEAM on 2026;
(b) the same five-term recalibration applied to PLAYER v1 was scored on 2026 in the BLEND study:
    P_RECAL 0.156654 vs TEAM 0.156797, 0.3-0.5 band gap +0.11 points (`RESULTS_BLEND_2026.md`), so
    the 2026 effect of the recalibration step on v1 is known;
(c) the engine's 2026 band gap (+3.26 points) — the reason for the recalibration;
(d) the bench-call study (`research/bench_claude/RESULTS.md`) found on 2026 that the teams' recent
    usage pattern matched managers' reliever choices better than the model's best reliever
    (28.7% vs 20.1%); the usage bullpen was built before that study was scored, but the finding
    supports the same idea.
The usage bullpen and the recalibration were designed and sized on 2022-2025 only, and no 2026
prediction of this arm has been compared with an outcome. The 2027 regular season is the clean check.

## Model (frozen)
- 2026 predictions: `size_pen.py --season 2026 --variant B --load-seasons 2021,...,2026 --part K/4`
  (K = 0..3), stored as `published/penB_2026_part{K}.parquet` (frozen below). Engine and class tables,
  league rates and the starter-length table fitted on 2021-2025 (in the sizing folds they were fitted
  on the other four of 2021-2025, including later seasons, and the record history before a season was
  1-4 years instead of 5; the recalibration was learned on those folds);
  batter / pitcher / bullpen / defense records and usage / rest tables from earlier dates only
  (usage and rest within 2026); park factors from the previous three seasons, average for OAK and TB
  (as `test_2026.py`; HF `park_factors.parquet` md5 3f1c06008ec1fec5012a8f50338b9687, checked by
  the script with the plate-appearance and pitch tables). Gates before the freeze: `size_pen.py` after
  the options for this test were added reproduces the earlier 40-game file of 2022 (max difference
  1.1e-16), and the same 40 games run as `--part 0/2` and `--part 1/2` give the same rows.
- Recalibration: `test_pen_2026.py fit` on the 2022-2025 out-of-fold predictions of `size_pen.py`
  variant B (733,260 rows) → `published/pen_recal_coef.json`. Gate: refit on 2022-2024 and scored on
  2025 (`dry`) gives Brier 0.152641, equal to `size_stack_pen.json` P_RECAL 2025.
- PEN_RECAL = sigmoid(design · coef) on the 2026 predictions (`test_pen_2026.py predict`); the score
  step recomputes it from the frozen parts and coefficients and stops if it differs.
- Rehearsal of the scoring path on 2025 sizing rows (`score_dry`, no 2026 read): PEN_RECAL 0.152641,
  PLAYER v1 0.154562, difference −0.00192 (95% −0.00251 to −0.00129).

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(PEN_RECAL) < Brier(TEAM) (`p_TEAM` of the frozen team layer, `research/wp_layer/published/pred_2026.parquet`).
- H2: Brier(PEN_RECAL) < Brier(PLAYER) (`wp_player`, `published/pred_2026_part*.parquet`).
One-sided, game bootstrap (2,000 resamples, seed 20261030), p = (1 + #{resampled difference ≥ 0}) /
2001, Holm over the two, supported if the adjusted p ≤ 0.05 (so H1's verdict can depend on H2's p).
Brier on unclipped probabilities.
Expected resolution: the 95% interval of a difference between two layers on 2026 had half-width
about 0.001 (v1 test H2); the sizing difference is about −0.0018, so both hypotheses can be
supported if the gain carries over, and a NOT SUPPORTED result is possible even if it is real.
Reported, not tested: log loss and the 0.3-0.5 band gap of PEN_RECAL, PEN (not recalibrated), TEAM,
PLAYER, ENGINE, BLEND.

## Run
`test_pen_2026.py predict` (no outcome read), then `score` once. The score step requires
Status: FROZEN and checks every md5 below and the input data md5s in the script (pa / pitch
2021-2026, park_factors).
Outputs `published/pen_2026.parquet`, `published/results_pen_2026.json`, `RESULTS_PEN_2026.md`.

## What follows
- H1 supported → the look-back view (finished games 2022-2026, where usage tables exist) shows
  PEN_RECAL as its win probability; live and the API keep TEAM until a 2027 check (live has no usage
  tables). The README states the disclosure above.
- H1 not supported → the headline stays TEAM; PEN_RECAL's numbers are published either way.

## Frozen files (md5)
0d61f77c584cc7ef414c9c198aa9e7ec  research/player_layer/test_pen_2026.py
d2170512c0acc37b918cdb4ce4b59d25  research/player_layer/size_pen.py
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
a71518eba9ad596861857508619bc270  research/player_layer/size_stack.py
12972ac8bb23a2dc8c152c07cbfd7a56  wp_engine.py
b979748de5cd4e160981485454a08dd7  research/wp_layer/layer_features.py
271cd82d1561a001920fa86d89f4f27f  research/player_layer/published/pen_recal_coef.json
17e139c538b2122272345dc81a69a1c9  research/player_layer/published/size_stack_pen.json
8e97bcc37f14d66f9cf165bf69e28746  research/player_layer/published/size_pen.json
d0329712d57441fb7e73f6bfd03513ac  research/player_layer/published/penB_2026_part0.parquet
fe6e4b7c0cfcbbbeab875c1f52e8c6f4  research/player_layer/published/penB_2026_part1.parquet
b281a3c1da82f56c7fbbf4b43e09d1fd  research/player_layer/published/penB_2026_part2.parquet
d255c1625a9380dca907da7e65d2d672  research/player_layer/published/penB_2026_part3.parquet
fb899b2b625b61b6220dc6389e528c57  research/player_layer/published/pred_2026_part0.parquet
370b161ae892e81b74f64630f906e575  research/player_layer/published/pred_2026_part1.parquet
9c0e9909b04f833e70b9bbfcd9e0e265  research/player_layer/published/pred_2026_part2.parquet
c65338b792b53f250f54ba7c39f9eba6  research/player_layer/published/pred_2026_part3.parquet
64f6c3f0cd90b5d6adcbefd7abb843ee  research/player_layer/published/blend_2026.parquet
c9e35870ab157190d2d81b1f26b23b87  research/wp_layer/published/pred_2026.parquet

## AMENDMENTS
(none)
