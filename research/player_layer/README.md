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

## Next
1. Whole game: carry the lineup and pitcher (starter → bullpen) into the
   remaining half-innings, add park (V5 + park).
2. Fielder-level defense from the real alignment.
3. Compare with the team layer, then freeze a PREREG and score 2026 once
   (2026 was already used for the team layer and the steal study).
