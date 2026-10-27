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

## Next
1. Compare with the team layer on the same rows (the team layer was
   fitted on 2021-2025, so only a 2026 comparison is fair), then freeze a
   PREREG (player layer v1 vs engine and vs team layer) and score 2026
   once; 2026 was already used for the team layer and the steal study.
2. Bench use: per-option run distributions (pinch hitter / pitcher change
   / bunt) with handedness, shown in the bench screen (task 3).
