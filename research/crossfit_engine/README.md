# Cross-fitted engine (leave one season out, 2021-2025)

Infrastructure for every new layer (CLAUDE.md next task 1). Measured in a
Claude Code cloud session on 2026-10-26; 2026 was not read.

```bash
python research/crossfit_engine/build_oof.py --states ~/claude-scratch/mlbwp/data/states --out ~/claude-scratch/crossfit
```
About 2 minutes. Writes `oof_wp.parquet` (915,418 PAs; md5
`f08b52c3b5af5bb6339c5dbc7122bb69` in that run, not committed: rebuild it)
and `summary.json` (committed as `published/summary.json`).

## Gate
Refitting `wp_engine.fit_tables` on all of 2021-2025 reproduces
`engine_tables.json`: `transitions`, `by_class`, `bunt`,
`class_given_count` and `n_pa` are equal. `league_swing` (a sampled mean,
not used by the script) differs in the last digit
(0.03429431701507597 vs 0.034294317015075966).

## Results (2021-2025, every PA)
| | Brier | log loss | WP 0.3-0.5: predicted − observed |
|---|---|---|---|
| out-of-fold engine | 0.157555 | 0.469009 | +2.61 pt (n 144,489) |
| published engine (in-sample) | 0.157531 | 0.468946 | +2.59 pt |

The 0.3-0.5 bias is positive out-of-fold in every season (2021 +1.99,
2022 +4.77, 2023 +1.61, 2024 +4.08, 2025 +0.66). It is not in-sample
overfitting: holding the season out leaves it in place, so a layer
fitted against `wp_oof` has a real error to learn.
