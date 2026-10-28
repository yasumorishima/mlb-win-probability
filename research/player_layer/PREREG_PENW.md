# Pre-registration: PENW_RECAL (PEN_RECAL with a 60-day reliever usage window) on 2026

Status: FROZEN (2026-10-28, cloud). Nothing below changes after the commit that set this line, except
in a dated AMENDMENTS section written before the score step is run. Independent review before the
freeze (read-only subagent, no 2026 outcome read): blocking points fixed — the comparison file
`pen_2026.parquet` pinned by md5 (checked against `results_pen_2026.json` at scoring), 2026 part file
names; suggestions taken: 2026-path gate, `wp_penw` recompute check, disclosures, narrower consequences.

## Why
PEN_RECAL (player layer + late-inning bullpen by usage, recalibrated; `PREREG_PEN.md`) is the look-back
view's win probability since it beat TEAM and PLAYER on 2026 (`RESULTS_PEN_2026.md`). Its bullpen
weights each reliever by his share of the team's relief plate appearances in that inning and lead state
over the previous **30** days. Sizing after that test, on 2022-2025 only (2026 not read for any of it;
`README.md` sections "Where the bullpen gain comes from" and "Usage window"):
- placebo (roles shuffled within each team): +0.00087 vs PEN_RECAL, pooled team bullpen +0.00173 —
  about half of the bullpen gain comes from who pitches which inning and score, so the role estimate
  matters;
- the same layer with usage counted over **60** days (`size_pen_x.py --window 60`), recalibrated out
  of fold the same way: −0.000093 vs PEN_RECAL (95% −0.000118 to −0.000068), every season negative
  (−0.000053 / −0.000105 / −0.000096 / −0.000117 in 2022 / 2023 / 2024 / 2025;
  `published/size_vs_pen_w60.json`); **120** days −0.000084 (−0.000120 to −0.000048), every season
  negative (`published/size_vs_pen_w120.json`). 60 was chosen because its 2022-2025 mean was lower.
  Windows tried: 30 (the tested layer), 60, 120 — no others.
Other candidates sized against PEN_RECAL on 2022-2025 and not taken: usage from the 4th inning,
times through the order (two forms), starter length by pitches thrown (README sections).
This test asks whether the 60-day window improves the look-back win probability on 2026.

## Not blind (disclosure)
2026 has been read for this repository's earlier tests (engine, team layer, steal study, player layer
v1, BLEND, bench-call study, PEN_RECAL). In particular the PEN_RECAL result on 2026 is known:
Brier 0.155063, vs TEAM −0.00173 (−0.00262 to −0.00076), vs PLAYER −0.00170, band gap +0.64 pt
(`published/results_pen_2026.json`). Its predictions are the comparison arm here, unchanged; its
per-play 2026 values are published (`pen_2026.parquet`, `pen_wp_2026.parquet`) and the look-back view
has shown them next to 2026 game results for single games. No 2026 prediction with any window other
than 30 days has been made or compared with an outcome.
Also known: the bench-call study found on 2026 that the 30-day usage rule matched managers' reliever
choices better than the model's best reliever (28.7% vs 20.1%, `research/bench_claude/RESULTS.md`).
Selection on 2022-2025: six candidates have been sized against PEN_RECAL (usage from the 4th inning,
two times-through-the-order forms, pitch count, 60 days, 120 days); 60 was picked over 120 on a mean
difference of about 0.000009, and the same seasons fitted the recalibration.

## Model (frozen)
- 2026 predictions: `size_pen_x.py --season 2026 --variant B --window 60 --load-seasons
  2021,2022,2023,2024,2025,2026 --part K/4` (K = 0..3), stored as
  `published/penB_w60_2026_part{K}.parquet` (frozen below). Everything else as PEN_RECAL's 2026
  predictions (tables fitted on 2021-2025, records / usage / rest from earlier dates only, usage within
  2026, same park rule). `size_pen_x.py` without options gives the same predictions as the frozen
  `size_pen.py` (gate, re-run on the version frozen below: the same 40 games of 2022, max difference 0 vs
  frozen `size_pen.py`, 1.1e-16 vs the earlier file; and the 2026 path: `--season 2026 --load-seasons
  2021,...,2026 --games 40` with the default window equals the matching rows of the frozen
  `penB_2026_part*` files; results in "Gates" below).
- What the window changes: only the usage counts (`usage_tables`); with 60 days the counts are about
  twice as large, so the shrinkage toward the pooled bullpen (KAPPA 30) and the lead-state smoothing
  (TAU 20) are weaker. Rest table, records, engine, starter table and park rule are unchanged.
- Recalibration: `test_penw_2026.py fit` on the 2022-2025 out-of-fold sizing predictions of the 60-day
  layer (733,260 rows) → `published/penw_recal_coef.json`; same terms as PEN_RECAL (L, L·r, L·r², r, r²).
- PENW_RECAL = sigmoid(design · coef) on the 2026 predictions (`test_penw_2026.py predict`); the score
  step recomputes it from the frozen parts and coefficients and stops if it differs.
- Rehearsal on 2025 (`dry`, both layers recalibrated on 2022-2024, no 2026 read): PENW_RECAL 0.152524,
  PEN_RECAL 0.152641 (= `size_stack_pen.json` P_RECAL 2025), difference −0.000117
  (95% −0.000165 to −0.000066).

## Arms and hypotheses (2026, 184,327 plate appearances, one run)
- H1: Brier(PENW_RECAL) < Brier(PEN_RECAL) (`p_PEN_RECAL` of `published/pen_2026.parquet`, the file
  scored in PREREG_PEN: md5 equal to `pred_md5` in `results_pen_2026.json`, checked by the score step,
  and frozen below).
- H2: Brier(PENW_RECAL) < Brier(TEAM) (`p_TEAM`, `research/wp_layer/published/pred_2026.parquet`).
  A consistency check, not new evidence: PEN_RECAL − TEAM on 2026 is known (−0.00173).
One-sided, game bootstrap (2,000 resamples, seed 20261030, `test_pen_2026.run_tests`),
p = (1 + #{resampled difference ≥ 0}) / 2001, Holm over the two, supported if the adjusted p ≤ 0.05.
Brier on unclipped probabilities.
Expected resolution: on 2025 the interval of PENW_RECAL − PEN_RECAL had half-width about 0.00005 and
the sizing difference is about −0.0001, so H1 can be supported if the gain carries over; NOT SUPPORTED
is possible even if it is real. H2 is expected to follow PEN_RECAL's known result.
Reported, not tested: log loss and the 0.3-0.5 band gap of PENW_RECAL, PEN_RECAL, TEAM, PENW (not
recalibrated).

## Run
`test_penw_2026.py predict --out research/player_layer/published` (no outcome read), then
`score --out research/player_layer/published` once. The score step requires
Status: FROZEN and checks every md5 below and the input data md5s (pa / pitch 2021-2026,
park_factors, via `test_pen_2026.check_inputs`).
Outputs `published/penw_2026.parquet`, `published/results_penw_2026.json`, `RESULTS_PENW_2026.md`.

## What follows
- H1 supported → the look-back view's win probability series becomes PENW_RECAL (2022-2026 games; the
  2022-2025 values from the sizing predictions recalibrated with `penw_recal_coef.json`, as
  `build_pen_wp.py` did for PEN_RECAL), and the bullpen mixture inside the bench options uses 60-day
  usage. The 「起用パターン」 pick (`bench_replay.usage_pick`, the bench-call study's rule) stays at
  30 days: nothing here tests a 60-day pick.
- H1 not supported → PEN_RECAL stays; the numbers are published either way.

## Gates (run before the freeze)
- `size_pen_x.py` (this version) without options vs frozen `size_pen.py`, 2022, 40 games: max difference 0
  (1.1e-16 vs the earlier file `penB_2022_g40`).
- 2026 path: `size_pen_x.py --season 2026 --variant B --load-seasons 2021,...,2026 --games 40` (30 days)
  vs the frozen `penB_2026_part*`: 3,092 rows, max difference 0.
- `dry` on 2025: PENW_RECAL 0.152524, PEN_RECAL 0.152641, −0.000117 (95% −0.000165 to −0.000066).
- `predict` written to `published/penw_2026.parquet` (184,327 rows, no outcome read).

## Frozen files (md5)
45a794a88aded87976ec4f5567f2be1d  research/player_layer/test_penw_2026.py
0d61f77c584cc7ef414c9c198aa9e7ec  research/player_layer/test_pen_2026.py
a71518eba9ad596861857508619bc270  research/player_layer/size_stack.py
b979748de5cd4e160981485454a08dd7  research/wp_layer/layer_features.py
9873dd9c9e0144a71d7c676c1b9940b8  research/player_layer/size_pen_x.py
d2170512c0acc37b918cdb4ce4b59d25  research/player_layer/size_pen.py
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
f8e0379dc4ef5e146fb02f3633ac60ae  research/player_layer/size_vs_pen.py
12972ac8bb23a2dc8c152c07cbfd7a56  wp_engine.py
4d218e568a0deb7af0272192eb847815  research/player_layer/published/penw_recal_coef.json
8257406c8c65d2e2c4349a6debf6fdf5  research/player_layer/published/size_vs_pen_w60.json
6524aac83fb642d0f9da5d4d0ee2d753  research/player_layer/published/size_vs_pen_w120.json
b62ca366ecd9ef80827587317982aa96  research/player_layer/published/size_vs_pen_placebo.json
9e298fd60cbdc9d8b50c269766bb9364  research/player_layer/published/size_vs_pen_pooled.json
dfc422d5dfd570a61ab02175874a79ab  research/player_layer/published/penB_w60_2026_part0.parquet
6faff0296673b7e1dea46f5c6f8b7192  research/player_layer/published/penB_w60_2026_part1.parquet
56c1728c7dd54497ec96004f1b2668f7  research/player_layer/published/penB_w60_2026_part2.parquet
dfc03a5860116868493d9eb65d9211df  research/player_layer/published/penB_w60_2026_part3.parquet
44a1af9d14fa053e8db177a0f048c0ac  research/player_layer/published/penw_2026.parquet
59c936efd04621678a0734f3cf3bb22d  research/player_layer/published/pen_2026.parquet
3383e06af583d8869132e765233f39d5  research/player_layer/published/results_pen_2026.json
c9e35870ab157190d2d81b1f26b23b87  research/wp_layer/published/pred_2026.parquet

## AMENDMENTS
(none)
