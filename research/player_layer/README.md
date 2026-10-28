# Player layer — sizing on 2021-2025 (work in progress)

Goal (owner, 2026-10-26): run expectancy and the chance of scoring depend
on the opposing pitcher, the defense and the own lineup; the layer gives
per-situation run distributions for the actual next batters, the base for
pinch-hit / pinch-run / pitching-change / bunt calls.

Everything below was measured in a Claude Code cloud session on
2026-10-26/27, on 2022-2025, each season scored with league tables and an
engine fitted on the other four of 2021-2025; player and team records use
only dates before the plate appearance. **2026 was not read.** These are
sizing numbers (parameters were chosen on the same seasons), not a test.

## Stage A — plate-appearance outcome (`pa_rates.py`, `size_pa.py`, `size_context.py`)
Classes K, BB, 1B, 2B, 3B, HR, OUT; multinomial log loss per PA
(240,000 + PAs per season). Paired differences vs league-only, 95% game
bootstrap.

| variant | log loss | vs league |
|---|---|---|
| V0 league by base-out state (the engine) | 1.45405 | |
| V1 × batter | 1.44036 | −0.0137 |
| V2 × batter × pitcher (log5) | 1.43494 | −0.0191 (−0.0195 to −0.0187) |
| V3 pitcher only on K/BB/HR/in-play total, fielding team on in-play outcomes | 1.43423 | −0.0198 |
| V5 as V3, batter shrinkage split (K/BB/HR/in-play 100, in-play split 400), pitcher 200 / 2000 | 1.43343 | −0.0206 |
| V5 + park (previous-season Savant class factors, half strength) | 1.43300 | −0.0004 vs V5 (−0.0005 to −0.0004) |

Defense: the fielding team's in-play outcomes earlier in the season
change the log loss by −0.0001; the previous season's team OAA by 0 (the
best weight on the grid is zero). Neither measures who is actually on the
field; fielder-level OAA with the game's real alignment (Stats API
boxscores) is the next thing to try.

## Stage B/C — rest of the half-inning and WP (`layer.py`, `size_runs.py`)
Dynamic programme over (base-out state, lineup slot) with the actual
lineup as known at the moment (last occupant of each slot; the next batter
is the actual one in 96.7% of PAs, the rest are later substitutions), the
current pitcher and the defense, through the engine's per-class
transitions. Gate: with all multipliers = 1 it reproduces the engine's
run distributions (max difference 4e-8) and the task-1 out-of-fold WP
(max 6e-16). Later half-innings stay league-average.

| 733,260 PAs | league | player | player − league (95%) |
|---|---|---|---|
| P(≥1 more run this half), log loss | 0.50491 | 0.50303 | −0.00188 (−0.00215 to −0.00164) |
| runs this half, squared error | 0.94597 | 0.94191 | −0.0041 (−0.0048 to −0.0032) |
| WP Brier | 0.157223 | 0.156935 | −0.00029 (−0.00035 to −0.00023) |
| WP 0.3-0.5: predicted − observed | +2.76 pt | +2.71 pt | |

P(≥1) improves in every season (2022 −0.0020, 2023 −0.0021, 2024 −0.0011,
2025 −0.0023). The 0.3-0.5 band bias does not move: only the current
half-inning is player-specific so far.

## Whole game (`game.py`, `size_game.py`)
Every remaining half-inning gets a team-specific run distribution: the
batting team's lineup as known now against the pitcher expected in that
inning (the fielding team's starter through the 6th if he is still in,
otherwise its bullpen = its relievers' earlier PAs this season, pooled),
park factors at half strength. The fielding team's next turn starts at
its known next slot; later turns average over the nine slots. The game
recursion follows the engine's rules; with league distributions it equals
`Engine._H` / `Engine.wp` exactly (difference 0 for 9- and 7-inning
games), and `wp_league` equals the task-1 out-of-fold WP (difference 0).

| 733,260 PAs, 2022-2025 | Brier | log loss | WP 0.3-0.5 bias |
|---|---|---|---|
| league (task 1 baseline) | 0.157223 | 0.467899 | +2.76 pt |
| player, current half only | 0.156935 | 0.467174 | +2.71 pt |
| player, whole game | **0.155197** | **0.462994** | **+1.84 pt** |

Whole game − league: −0.00203 Brier (95% game bootstrap −0.00277 to
−0.00127); every season negative (2022 −0.0030, 2023 −0.0018, 2024
−0.0023, 2025 −0.0011). Whole game − current half: −0.00174 (−0.00246 to
−0.00104). For scale, the team layer's pre-registered 2026 gain was
−0.00151 vs its recalibration control (a different season and baseline:
not comparable as a test). Calibration by tenths is in
`published/size_game.json`; predictions below 0.5 are still too high
(0.1-0.2: 0.150 predicted vs 0.133 observed).

## Second round (2026-10-27/28): handedness, starter length, fielders
Same folds, 2022-2025, 2026 not read. Stats API data fetched with
`fetch_context.py` (handedness of 2,944 players; starting fielders of
every 2021-2025 game, 8.0 per team-game) into scratch, not committed.

Plate-appearance level (multinomial log loss, vs V5 + park):
| term | change (95%) |
|---|---|
| league platoon factor (batter side x pitcher hand) ** 0.5 | −0.00034 (−0.00039 to −0.00030) |
| + own splits of batter and pitcher (k = 2000) | −0.00064 (−0.00071 to −0.00058) |
| starting fielders' previous-season OAA (total, or infield / outfield) | 0 (best weight on the grid is 0; 57% of fielders have a previous-season OAA) |

Win probability (733,260 PAs, Brier, vs the whole-game layer above = v1):
| variant | Brier | vs v1 (95%) | 0.3-0.5 bias |
|---|---|---|---|
| v1 (starter through the 6th) | 0.155197 | | +1.84 pt |
| starter: P(starter begins inning i \| inning now, his batters faced) from the training folds, starter / bullpen mixed | 0.155168 | −0.00003 (−0.00008 to +0.00002) | +1.91 pt |
| starter + handedness | 0.155213 | +0.00002 (−0.00004 to +0.00007) | +1.80 pt |

Reading: handedness sharpens the plate-appearance prediction (what a
pinch-hit or pitching-change comparison uses) but does not change the
game-level Brier; the starter-length model does not either. Neither
interval excludes 0, so v1 stays the candidate for the win-probability
test; handedness stays in the plate-appearance part for bench
comparisons. Fielder-level OAA gave nothing measurable here; this is a
null for this measure (previous-season OAA, starting alignment), not a
statement that defense does not matter.

## Late-inning bullpen by usage (2026-10-28, `size_pen.py`, `game_pen.py`)
Same folds, 2022-2025, 2026 not read; compared on the same 733,260 rows with
the second round's "starter" variant (starter-length model, pooled bullpen).
Only the bullpen in innings 7+ changes. For each team and inning (7, 8, 9;
10+ uses 7-9 together), who pitches is a mixture of its relievers weighted
by their share of the team's relief plate appearances in that inning over
the 30 days before the game (same season), without pitchers already used in
the game (including the current one) and anyone who pitched on both
previous days, shrunk to the pooled bullpen with weight 30 / (30 + usage
PAs). Variant B also splits the usage by the fielding team's lead when the
half begins (behind / tied / ahead 1-3 / ahead 4+; 20 pseudo-PAs of the
inning share), so the half's run distribution depends on the score
(`game_pen.py`: game.py's recursion with score-dependent distributions;
gate: identical to game.py when they do not depend on the score, max
difference 0). Gate for the whole script: variant N (pool weight 1)
reproduces the "starter" predictions (max difference 1e-15, 40 games).

| 733,260 PAs, 2022-2025 | Brier | vs starter (95% game bootstrap) | log loss | 0.3-0.5 bias |
|---|---|---|---|---|
| starter (pooled bullpen) | 0.155168 | | 0.462926 | +1.91 pt |
| A: usage by inning | 0.153875 | −0.00129 (−0.00152 to −0.00106) | | +2.64 pt |
| B: usage by inning and lead | **0.153533** | **−0.00163 (−0.00187 to −0.00141)** | **0.459171** | +2.68 pt |

B minus starter by season: 2022 −0.00125, 2023 −0.00160, 2024 −0.00175,
2025 −0.00194; plate appearances in innings 7+ only: −0.00083 (−0.00100 to
−0.00065). B separates wins better but its probabilities are less extreme
than the outcomes in the middle (0.1-0.5: predicted 2-3 points above the
observed rate; 0.6-0.9: 0.8-1.5 below; table in
`published/size_pen.json`), so the 0.3-0.5 band bias grows; an
out-of-fold recalibration (as in the BLEND sizing) is the known fix and is
not applied here. This is sizing, not a test: B is now the bullpen in the
bench options (`bench.py` with `pen_avail_*`, usage tables
`published/bullpen_usage_<season>.json` from `build_usage.py`, the same
rules in `pen_usage.py`, equal to size_pen.py on 300 sampled team-days).
In the bench, a pitching change also removes that reliever from the later
innings (the cost of using the closer early). Pinch-hit / pinch-run options
use the pooled bullpen for their lineup plus the mixture's difference from
the pool measured with the current lineup; on a recorded 2025 game (108
options, 7th inning or later) that differs from the exact mixture by at
most 0.06 WP points (`bench_pen_check.py`).

## Relievers by usage from the 4th inning (sizing 2022-2025, 2026 not read)
`size_pen_x.py --first-inning 4`: the usage mixture (same rules as variant B: share of the team's relief
plate appearances in that inning and lead state over 30 days, pool weight from the usage of innings
4-6 for innings 4-6) replaces the pooled bullpen from the 4th inning instead of the 7th. Both layers
recalibrated out of fold the same way as PEN_RECAL (`size_vs_pen.py`, `published/size_vs_pen_f4.json`).
Gate: with the default (7th) the 40-game file of 2022 is reproduced (1.1e-16).

| | Brier (recalibrated) | 0.3-0.5 band |
|---|---|---|
| PEN_RECAL (from the 7th) | 0.153362 | +0.55 pt |
| from the 4th | 0.153342 | +0.55 pt |

Difference −0.00002 (95% −0.00006 to +0.00002); by season
2022 −0.00005, 2023 +0.00004, 2024 −0.00003, 2025 −0.00004. Not better in every season and the interval
includes 0: no gain at this size, so it is not taken further (PEN_RECAL stays).

## Times through the order (sizing 2022-2025, 2026 not read)
PA level (`size_tto.py`, `published/size_tto.json`): class factors by the starter's time through the order
(1, 2, 3, 4+) and for relievers, fitted on the other four of 2021-2025, on top of the player layer's PA
probabilities: log loss −0.00071 (95% −0.00080 to −0.00061), every season; largest the third
time through (−0.00270; strikeouts about 13% fewer, extra-base hits more).

WP level (`size_pen_x.py --tto`, recalibrated out of fold as PEN_RECAL, `size_vs_pen.py`; gate: without
`--tto` the 40-game file of 2022 is reproduced, 1.1e-16):

| | Brier (recalibrated) | minus PEN_RECAL (95%) | by season 2022-2025 |
|---|---|---|---|
| PEN_RECAL | 0.153362 | | |
| TTO, later innings at the expected later time through | 0.153458 | +0.00010 (+0.00006 to +0.00013) | +0.00014, +0.00009, +0.00008, +0.00008 |
| TTO in the current half only | 0.153438 | +0.00008 (+0.00004 to +0.00011) | +0.00011, +0.00008, +0.00005, +0.00006 |

Both are **worse** in every season. The PA-level gain does not carry over to the win probability; a
likely reason is selection: a starter who is still pitching later in the game is having a good day,
which the starter-length model already reflects, so a penalty for his later times through overstates
it. Not taken further (PEN_RECAL stays).

## Starter length by pitch count (sizing 2022-2025, 2026 not read)
`size_pen_x.py --pc-edges 0,30,45,60,75,85,95,105,1000` (`starter_pc.py`): the chance that the starter begins
a later inning is looked up by his pitches thrown so far (8 buckets, edges fixed before the run) instead of
batters faced (5 buckets); everything else as PEN. Pitches come from `pitch_<season>.parquet` (all pitches
of a plate appearance are credited to the pitcher on record). Gates: the table with batters faced in place
of pitches equals `size_game2.starter_table` (diff 0); without the option `size_pen_x.py` equals the frozen
`size_pen.py` on the 40-game file of 2022 (diff 0). Recalibrated out of fold as PEN_RECAL (`size_vs_pen.py`,
`published/size_vs_pen_pc.json`):

| | Brier (recalibrated) | minus PEN_RECAL (95%) | by season 2022-2025 |
|---|---|---|---|
| PEN_RECAL (batters faced) | 0.153362 | | |
| pitches thrown | 0.153347 | -0.000015 (-0.000033 to +0.000004) | -0.000017, +0.000009, -0.000024, -0.000028 |

Not better in every season (2023 worse) and the interval includes 0: no gain at this size, not taken
further (PEN_RECAL stays). Band 0.3-0.5 unchanged (+0.55 pt).

## Where the bullpen gain comes from: placebo (diagnostic, 2022-2025, 2026 not read)
`size_pen_x.py --placebo`: each team's usage rows (relief plate appearances by inning and lead state over 30
days) are handed to its own relievers in a random order (seeded by team and date), so the team's recent
relief corps, its total usage and the availability rules (used / pitched both previous days) stay, but who
pitches which inning and score state is random. Compared with the pooled team bullpen (size_game2 "starter",
the baseline before usage). All recalibrated out of fold as PEN_RECAL (`size_vs_pen.py`,
`published/size_vs_pen_placebo.json`, `published/size_vs_pen_pooled.json`):

| | Brier (recalibrated) | minus PEN_RECAL (95%) | by season 2022-2025 | 0.3-0.5 band |
|---|---|---|---|---|
| PEN_RECAL (usage by role) | 0.153362 | | | +0.55 pt |
| roles shuffled within the team (placebo) | 0.154229 | +0.00087 (+0.00079 to +0.00094) | +0.00072, +0.00087, +0.00102, +0.00086 | +0.38 pt |
| pooled team bullpen (no usage) | 0.155096 | +0.00173 (+0.00150 to +0.00196) | +0.00150, +0.00158, +0.00186, +0.00200 | +0.33 pt |

About half of the usage layer's gain over the pooled bullpen comes from roles (who pitches the late innings
at each score), the other half from the recent relief corps and availability; both parts are positive in
every season. This is a diagnostic, not a candidate.

## Usage window (sizing 2022-2025, 2026 not read)
`size_pen_x.py --window N`: reliever usage counted over the previous N days instead of 30 (same season,
same rules otherwise). Recalibrated out of fold as PEN_RECAL (`size_vs_pen.py`). Windows tried: 30, 60, 120.

| | Brier (recalibrated) | minus PEN_RECAL (95%) | by season 2022-2025 | 0.3-0.5 band |
|---|---|---|---|---|
| 30 days (PEN_RECAL) | 0.153362 | | | +0.55 pt |
| 60 days | 0.153269 | -0.000093 (-0.000118 to -0.000068) | -0.000053, -0.000105, -0.000096, -0.000117 | +0.53 pt |
| 120 days | 0.153278 | -0.000084 (-0.000120 to -0.000048) | -0.000039, -0.000101, -0.000110, -0.000086 | +0.52 pt |

Both longer windows are better in every season; 60 days has the lower mean and goes to a pre-registered
test on 2026 against PEN_RECAL (`PREREG_PENW.md`).

## Next
1. Compare with the team layer on the same rows (the team layer was
   fitted on 2021-2025, so only a 2026 comparison is fair), then freeze a
   PREREG (player layer v1 vs engine and vs team layer) and score 2026
   once; 2026 was already used for the team layer and the steal study.
2. Bench use: per-option run distributions (pinch hitter / pitcher change
   / bunt) with handedness, shown in the bench screen (task 3).
