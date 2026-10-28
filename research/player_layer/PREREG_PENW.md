# Pre-registration: PENW_RECAL (PEN_RECAL with a 60-day reliever usage window) on 2026

Status: DRAFT (not frozen; nothing has been scored).

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
(filled before the freeze)

## Frozen files (md5)
(filled at the freeze)

## AMENDMENTS
(none)
