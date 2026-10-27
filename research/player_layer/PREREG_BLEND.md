# Pre-registration: BLEND (player layer + team strength) on 2026

Status: FROZEN (2026-10-28, cloud). Nothing below changes after the commit
that set this line, except in a dated AMENDMENTS section written before the
score step is run. Independent check before freezing: the blend recomputed
from `blend_coef.json` with separately written formulas reproduces the
fitted in-sample Brier on 2022-2025 exactly (0.15470736138005825).

## Why
The player layer (PLAYER, `RESULTS_2026.md`) and the team layer (TEAM,
`research/wp_layer`) each beat the engine on 2026 and could not be told
apart. On the 2022-2025 out-of-fold rows (`size_stack.py`,
`published/size_stack.json` / `size_stack_v2.json`) a blend of the two,
each recalibrated on out-of-fold predictions, had a lower Brier than the
player layer alone: −0.00036 (95% −0.00063 to −0.00008), the blend weight
0.5-0.55 in every fold, and the recalibration removed most of the 0.3-0.5
band bias (+1.84 → +0.25 points for the recalibrated player layer).
This test asks whether the blend is better than each of the two layers on
2026, to decide which number the app shows as the headline win probability.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests: the engine's
calibration, the team layer (TEAM − RECAL −0.00151), the steal study and
the player layer (PLAYER − ENGINE −0.00169, PLAYER − TEAM −0.00003).
**I knew before designing this test that PLAYER and TEAM are about equal
on 2026**, and blends of two about-equal forecasts usually score better
than either. The blend's form and weight were chosen on 2022-2025 only
(`fit_blend.py`, `published/blend_coef.json`), but the decision to test a
blend was made with that knowledge. A supported result is therefore weaker
evidence than an unseen season would be; the 2027 season is the clean
check.

## Model (frozen: `fit_blend.py` → `published/blend_coef.json`)
- P_RECAL: logistic on logit(PLAYER) with terms L, L·r, L·r², r, r²
  (r = share of regulation outs still to be played, floored at one half).
- E_TEAM: logistic on logit(ENGINE) with the same terms plus d_prev·r,
  d_prev·r², d_cur·r, d_cur·r² (team run differential per game, previous
  season and this season on earlier dates shrunk n/(n+30), as in
  `research/wp_layer/layer_features.py`).
- BLEND = sigmoid(0.5 · logit(P_RECAL) + 0.5 · logit(E_TEAM)).
- Coefficients and the weight fitted on the 2022-2025 out-of-fold
  predictions of `size_game.py` (733,260 rows); 2026 not read.
- 2026 inputs: the frozen player-layer predictions
  (`published/pred_2026_part*.parquet`: `wp_player`, `wp_engine`) and
  team strength from games on earlier dates.

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(BLEND) < Brier(TEAM) (`p_TEAM` of the frozen team layer).
- H2: Brier(BLEND) < Brier(PLAYER) (`wp_player`).
One-sided, game bootstrap (2,000 resamples, seed 20261029), p = (1 +
#{resampled difference ≥ 0}) / 2001, Holm over the two, supported if the
adjusted p ≤ 0.05. Brier on unclipped probabilities. Expected resolution:
the sizing interval for the blend's gain over the player layer on four
seasons had half-width about 0.00028; on one season about 0.0006, larger
than the sizing's mean gain (0.00036), so a NOT SUPPORTED result is likely
even if the gain is real and is not evidence of no gain.
Reported, not tested: log loss and the 0.3-0.5 band gap of BLEND, TEAM,
PLAYER, ENGINE, P_RECAL, E_TEAM.

## Run
`test_blend_2026.py predict` (no outcome read) after this file is frozen,
then `score` once. The score step requires Status: FROZEN and checks every
md5 below. Outputs `published/blend_2026.parquet`,
`published/results_blend_2026.json`, `RESULTS_BLEND_2026.md`.

## What follows
- H1 and H2 supported → the app's headline WP becomes BLEND (engine + team
  layer + player layer), with the disclosure above.
- Otherwise the headline stays the team layer; BLEND's numbers are
  published either way.

## Frozen files (md5)
680485814de0674a193c629f77067d32  research/player_layer/test_blend_2026.py
968b2751fc5ac862ef375fd3faa7b42d  research/player_layer/fit_blend.py
a71518eba9ad596861857508619bc270  research/player_layer/size_stack.py
d55c7aceb0d362f20a665e19533981de  research/player_layer/published/blend_coef.json
fb899b2b625b61b6220dc6389e528c57  research/player_layer/published/pred_2026_part0.parquet
370b161ae892e81b74f64630f906e575  research/player_layer/published/pred_2026_part1.parquet
9c0e9909b04f833e70b9bbfcd9e0e265  research/player_layer/published/pred_2026_part2.parquet
c65338b792b53f250f54ba7c39f9eba6  research/player_layer/published/pred_2026_part3.parquet
b979748de5cd4e160981485454a08dd7  research/wp_layer/layer_features.py

## AMENDMENTS
(none)
