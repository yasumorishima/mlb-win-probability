# Pre-registration: PEN_RECAL (player layer + late-inning bullpen by usage, recalibrated) on 2026

Status: DRAFT (becomes FROZEN in the commit that sets this line; nothing below changes after that,
except in a dated AMENDMENTS section written before the score step is run).

## Why
The headline win probability of the app is the team layer (TEAM, `research/wp_layer`). The player
layer v1 (PLAYER) beat the engine on 2026 but could not be told apart from TEAM
(−0.00003, 95% −0.00098 to +0.00097, `RESULTS_2026.md`), and BLEND did not beat either
(`RESULTS_BLEND_2026.md`). Since then, on 2022-2025 only (2026 not read):
- the late-inning bullpen by usage (`size_pen.py`, variant B: relievers weighted by their share of the
  team's relief plate appearances in that inning and lead state over the previous 30 days, without
  pitchers already used or tired) lowered the Brier of the player layer by 0.00163
  (95% −0.00187 to −0.00141) vs the same layer with a pooled bullpen, every season negative
  (`published/size_pen.json`);
- a logistic recalibration of that layer on out-of-fold predictions (terms L, L·r, L·r², r, r²,
  as `size_stack.py` P_RECAL) removed most of its 0.3-0.5 band bias (+2.68 → +0.55 points) and gave
  Brier 0.153362 vs 0.155185 for the engine with the same team-strength terms as TEAM (E_TEAM):
  −0.0018 overall, −0.00191 / −0.00153 / −0.00208 / −0.00179 in 2022 / 2023 / 2024 / 2025
  (`published/size_stack_pen.json`; the blend weight on the player side was 1.0 in every fold, so
  no blend is tested).
This test asks whether PEN_RECAL beats TEAM and PLAYER on 2026, to decide the headline number.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests (engine calibration, team layer, steal study,
player layer v1, BLEND, and the bench-call agreement study, which used 2026 manager decisions).
I knew that PLAYER ≈ TEAM on 2026. The usage bullpen and the recalibration were designed and sized
on 2022-2025 only, and no 2026 prediction of this arm has been compared with an outcome. The 2027
regular season is the clean check.

## Model (frozen)
- 2026 predictions: `size_pen.py --season 2026 --variant B --load-seasons 2021,...,2026 --part K/4`
  (K = 0..3). Engine and class tables, league rates and the starter-length table fitted on 2021-2025;
  batter / pitcher / bullpen / defense records and usage / rest tables from earlier dates only
  (usage and rest within 2026); park factors from the previous three seasons, average for OAK and TB
  (as `test_2026.py`). Gate before the freeze: `size_pen.py` after the options for this test were
  added reproduces the earlier 40-game file of 2022 (max difference 1.1e-16).
- Recalibration: `test_pen_2026.py fit` on the 2022-2025 out-of-fold predictions of `size_pen.py`
  variant B (733,260 rows) → `published/pen_recal_coef.json`. Gate: refit on 2022-2024 and scored on
  2025 (`dry`) gives Brier 0.152641, equal to `size_stack_pen.json` P_RECAL 2025.
- PEN_RECAL = sigmoid(design · coef) on the 2026 predictions (`test_pen_2026.py predict`).

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(PEN_RECAL) < Brier(TEAM) (`p_TEAM` of the frozen team layer, `research/wp_layer/published/pred_2026.parquet`).
- H2: Brier(PEN_RECAL) < Brier(PLAYER) (`wp_player`, `published/pred_2026_part*.parquet`).
One-sided, game bootstrap (2,000 resamples, seed 20261030), p = (1 + #{resampled difference ≥ 0}) /
2001, Holm over the two, supported if the adjusted p ≤ 0.05. Brier on unclipped probabilities.
Expected resolution: the 95% interval of a difference between two layers on 2026 had half-width
about 0.001 (v1 test H2); the sizing difference is about −0.0018, so both hypotheses can be
supported if the gain carries over, and a NOT SUPPORTED result is possible even if it is real.
Reported, not tested: log loss and the 0.3-0.5 band gap of PEN_RECAL, PEN (not recalibrated), TEAM,
PLAYER, ENGINE, BLEND.

## Run
`test_pen_2026.py predict` (no outcome read), then `score` once. The score step requires
Status: FROZEN and checks every md5 below and the input data md5s in the script.
Outputs `published/pen_2026.parquet`, `published/results_pen_2026.json`, `RESULTS_PEN_2026.md`.

## What follows
- H1 supported → the look-back view (finished games 2022-2026, where usage tables exist) shows
  PEN_RECAL as its win probability; live and the API keep TEAM until a 2027 check (live has no usage
  tables). The README states the disclosure above.
- H1 not supported → the headline stays TEAM; PEN_RECAL's numbers are published either way.

## Frozen files (md5)
