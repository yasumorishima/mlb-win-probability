# A correction layer for the engine: results on 2026

The study was pre-registered in [`PREREG.md`](PREREG.md) (frozen in commit
`0cc376d` before the run). `run_2026.py` was run on 2026-10-16 at 01:20 JST
and finished without a restart. This is the only run of the frozen script on
2026 (the frozen script was also run with `--dry` on the training seasons). An
earlier, unfrozen version had been run on 2026 by mistake on 2026-10-15 and
its output was deleted unread; [`PREREG.md`](PREREG.md) discloses it. The output of the frozen run is in [`published/`](published/):
`results_2026.json` (every number below), `pred_2026.parquet` (each arm's
prediction for every plate appearance) and `run_2026.log`. The input hashes
it recorded match `FROZEN_MD5`.

## Verdict

**H1 and H2 are SUPPORTED.** Knowing the two teams' run differentials (last
season and this season so far) makes the home win probability more accurate
than the engine alone and than a plain recalibration of the engine.

| hypothesis | difference in Brier score | 95% interval | one-sided p | Holm threshold | verdict |
|---|---:|---|---:|---:|---|
| H1 TEAM − RECAL | −0.00151 | −0.00253 to −0.00049 | 0.0035 | 0.05 | SUPPORTED |
| H2 TEAM − ENGINE | −0.00165 | −0.00270 to −0.00060 | 0.0030 | 0.025 | SUPPORTED |

2026: 184,327 plate appearances in 2,429 games. The interval comes from 2,000
bootstrap resamples of games.

An independent recomputation from `pred_2026.parquet`
([`check_independent.py`](check_independent.py), output in
`published/check_independent.log`, written after the run) used a different
implementation, seed and 4,000 resamples. It gives the same Brier scores to
five decimals and the same conclusion: TEAM − RECAL is −0.00151 (−0.00262 to
−0.00045, p 0.0037), and TEAM − ENGINE is −0.00165 (−0.00278 to −0.00056, p 0.0030).

The gain is about the size the training-season sizing expected (−0.0012 to
−0.0024 by season). The dry run, fitted on 2022–2024 and scored on 2025,
gave −0.00122 there, with an interval that crossed zero (−0.00253 to
+0.00009).

## All arms (reported)

| arm | Brier | log loss | Brier, early third | middle third | last third |
|---|---:|---:|---:|---:|---:|
| ENGINE | 0.15845 | 0.47138 | 0.21597 | 0.15653 | 0.09396 |
| RECAL | 0.15831 | 0.47115 | 0.21564 | 0.15639 | 0.09404 |
| **TEAM** | **0.15680** | **0.46743** | 0.21306 | 0.15514 | 0.09348 |
| SP | 0.15785 | 0.47006 | | | |
| INGAME | 0.15830 | 0.47112 | | | |
| FULL | 0.15666 | 0.46710 | | | |

As in the sizing, the point estimates are largest early in the game. TEAM −
RECAL is −0.0026 in the early third, −0.0013 in the middle third and −0.0006
in the last third. These are reported only, without intervals.

Reported-only differences (no claim is made from them):
- RECAL − ENGINE: −0.00014 (−0.00046 to +0.00019).
- SP − RECAL: −0.00047 (−0.00107 to +0.00020).
- INGAME − RECAL: −0.00001 (−0.00014 to +0.00013).
- FULL − TEAM: −0.00014 (−0.00038 to +0.00010).
- FULL − RECAL: −0.00165 (−0.00268 to −0.00062).

Starter quality and the in-game score terms add nothing that can be told
apart from zero once team strength is in.

## The 0.3–0.5 band (H3, reported with its interval, no verdict)

On the plate appearances where the engine says 0.3 ≤ p < 0.5, the engine's
mean prediction is 3.3 points above the observed home win rate. RECAL's mean
prediction there is 1.3 points above it and TEAM's 1.4 points. The difference
|gap TEAM| − |gap ENGINE| is −0.019 (−0.021 to +0.006).

| rows where the engine says 0.3–0.5 (28,814) | mean prediction | gap to observed 0.3705 |
|---|---:|---:|
| ENGINE | 0.4032 | +0.0326 |
| RECAL | 0.3833 | +0.0127 |
| TEAM | 0.3843 | +0.0137 |

The band's observed rate is itself uncertain (ENGINE's gap +0.033, 95%
interval +0.005 to +0.059 in `check_independent.log`). Recalibration lowers
the band's mean prediction by a tight amount, −0.0199 (−0.0200 to −0.0198),
but how much of the gap that closes depends on the observed rate and cannot
be told from one season (|gap RECAL| − |gap ENGINE| is −0.020; its lower end
is bounded by the shift itself and its upper end is +0.010). What is tight is
that team strength
does not move the band beyond what recalibration does: TEAM − RECAL in the
band's mean prediction is +0.001 (−0.0016 to +0.0036). So this study does not
show that team strength is what makes the engine too high in the band. The
README keeps the band caveat for the engine alone.

In fixed 0.1-wide bins of its own prediction, TEAM's mean prediction is
within 1.4 points of the observed rate in every bin (largest: 0.8–0.9,
predicted 0.852, observed 0.838).

## What follows (as pre-registered)

H1 is SUPPORTED, so the TEAM layer goes into the app and the API. It uses
the coefficients of this run, fitted on 2022–2025 and not refitted:

`L 1.0992, L·r −0.0533, L·r² 0.0308, r −0.0627, r² 0.0462,
d_prev·r 0.2762, d_prev·r² −0.1710, d_cur·r 0.3887, d_cur·r² −0.1577`
(raw units, from `results_2026.json`). The run did not record the
intercept. To ship the layer, the TEAM fit on 2022–2025 is redone with the
frozen code on the same data. The refit must reproduce these nine
coefficients, and the intercept is taken from it. This is a reproduction,
not a new fit.

`d_prev` and `d_cur` are defined exactly as in `layer_features.py`. The app
computes them from MLB Stats API results. Before the layer is shown, the
app's values are checked against `layer_features.py` on every 2025 game; any
difference above 0.01 runs per game stops the release.
