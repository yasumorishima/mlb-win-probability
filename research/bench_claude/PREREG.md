# Pre-registration — Claude's bench calls vs the managers' (pitching changes, pinch hitters)

Status: written 2026-10-27 (UTC) before any Claude answer on the test items was read and before
any agreement rate was computed for any arm. Frozen files and their md5 are in "Frozen" below;
the commit that fills in that section is the freeze (earlier commits of this file are drafts). Gaps found before the run go into "Amendments"
(a separate commit, before the run). Run once.

## Question
When an MLB manager has already decided to make a move — bring in a reliever (6th inning or later)
or send up a pinch hitter — does Claude pick the same player the manager picked more often than
(a) this app's player-layer model and (b) the team's recent usage pattern?

This measures **agreement with real managers**, not whether a call was right: the result of a call
that was not made cannot be observed. A higher agreement means Claude's choices look more like what
MLB managers actually do; it does not mean they are better.

## Data
- Games: a seeded random sample of 80 regular-season games played 2026-07-01..2026-09-27
  (`build_items.py`, seed 20261028, from the game list of `pa_2026.parquet`, Release `states-v1`).
  Pilot: 4 games of 2026-06-01..06-30 (seed + 1), never scored.
- Items: every pitching change from the 6th inning on (excluding the starting pitcher's entry), and
  every pinch hitter (substitution marked PH / "Pinch-hitter"), in those games. The moment is the
  one the call was made at, rebuilt from the game feed with no hindsight (`bench_replay.moment_at`
  with `upto_event`: lineups and bench through earlier substitutions, including those earlier in the
  same plate appearance — e.g. a pinch hitter sent up after a pitching change faces the new pitcher —,
  pitchers already used, player records = MLB totals 2021-2025, bullpen usage over the 30 days
  before the game date; game 1 of a doubleheader is not visible in game 2's usage / rest, for every arm).
- Candidates: the relievers not yet used that the bench screen lists (pitchers by primary position
  P / two-way; role: mostly relief work in 2024-2025, unknown kept), or the bench players not yet
  used (non-pitchers). Listed in id order. A position player who pitched in a blowout is therefore
  not a candidate, and such an item is excluded (label not among the candidates).
- Label: the player the manager sent in. Items whose label is not among the candidates (e.g. a
  starter used in relief) or with fewer than 2 candidates are excluded; their count is reported.
- Counts (read before freezing, needed to plan cost): 620 items in 80 games; 33 excluded
  (30 label not among the candidates, 3 fewer than 2 candidates); 422 pitching changes and
  165 pinch hitters scored. No agreement rate of any arm on these items was computed.

## Arms
- **C** — Claude (`claude-opus-5-5`, effort `medium`, structured output `{choice, top3, reason}`)
  with the moment, the candidates' records, recent usage / rest (pitchers), handedness, and the
  model's value of every candidate (WP change for the deciding team vs no change, and its rank).
- **B** — the same without the model's numbers.
- **MODEL** — the candidate with the highest model value (`bench.py`, player layer with the
  late-inning usage bullpen, as in the app; values rounded to 1e-4 points, ties: lowest id).
- **USAGE** — pitching changes only: the available candidate with the largest usage weight for that
  inning and the fielding team's lead when the half began (`pen_usage.weights`, variant B, the usage
  tables' definition; the 6th inning and extra innings use innings 7-9 together; ties: lowest id;
  no candidate with usage = miss).
- Same system prompt for B and C (`run_claude.py`). One answer per item and arm; no repeats.
  No server-side refusal fallback (the Batches API rejects it).

## Metric and inference
- Agreement = share of items where the arm's pick equals the manager's. An answer that cannot be
  parsed, is refused, is cut by max_tokens, or names a non-candidate counts as a miss (reported).
- Paired differences between arms on the same items; 95% interval and two-sided p from a bootstrap
  over games (10,000 resamples, seed 20261028).
- **H1**: on pitching changes, agreement(C) > agreement(MODEL).
- **H2**: on pitching changes, agreement(C) > agreement(USAGE).
- Holm over H1 and H2. A hypothesis is supported if its Holm-adjusted p < 0.05 and the difference is
  positive. Otherwise it is not supported (reported as a fact, with the interval).
- Secondary, no hypothesis (reported with intervals, never re-labelled as a test): B vs MODEL,
  B vs USAGE, C vs B, top-3 agreement (Claude's top3, the model's top 3), pinch hitters (C, B, MODEL),
  pitching changes by inning (6-7 vs 8+), chance level (mean 1 / number of candidates).
- Expected resolution: with about 450 pitching-change items and arms that disagree on 30-50% of
  items, the 95% half-width of a difference is about 4-6 points; smaller differences will read as
  "not supported".

## Disclosure
- The 2026 season was used before in this repository (team layer, steal study, player-layer test).
  Manager decisions of these games were not analysed before.
- Claude's training data end before July 2026 as far as we know; it may still know the players'
  roles before then (e.g. who closed in early 2026). That is part of its baseball knowledge, as a
  bench coach's would be.
- The player-layer model and the usage tables were built on 2021-2025 plus 2026 games before each
  game date (usage), not on these labels.

## Cost cap
- Pilot: the 23 non-excluded pilot items x arms B and C, live calls (46 calls, full price).
- The main run is two Batches (B, C) over the 587 test items (1,174 calls). Cap: the pilot's mean
  tokens per call priced at the Batches rate (half of $4 in / $20 out per MTok) x 1,174 must not
  exceed **$40**; otherwise stop and report (owner approved $60 for Claude studies).
  `run_claude.py pilot_cost` computes it.
- A request that errors in the batch is resubmitted once in a retry batch (`submit --retry`,
  `collect --retry`, appended; the last record per item counts); still failing = miss.
- Calls, tokens and cost are recorded in the results.

## Review before the freeze
An independent review (subagent, read-only, no API call, no agreement computed) found two blocking
problems in the draft, both fixed before this freeze and verified on the pilot items:
(1) a position player who pitched later in the game was a reliever candidate (and could be the
label) because the end-of-game boxscore position was used — fixed with the primary position;
(2) a call made after another substitution in the same plate appearance was built on the moment
before that substitution — fixed with `upto_event`. Smaller points also fixed: a retry path for
errored batch requests, the cost-cap pricing, the USAGE lead state (half start, as the tables),
ties, a resumable pilot. Tests in `tests/test_bench_replay.py` catch both blocking defects
(planted and caught).

## Outputs
`published/results.json` (analyze.py), `published/raw/` (all answers, gzipped), `RESULTS.md`
(numbers generated from results.json).

## Frozen
Paths relative to `research/bench_claude/`. Built and committed at `d1f9a96` (the items embed the
moments, candidates, labels, MODEL / USAGE picks and both payloads, so the app code may change later
without changing what is scored). Check with `md5sum -c` before the pilot and before analyze.

```
ff122e73e0c57a9e161ea0878d75c1ff  build_items.py
bbc312c96bd0359d25a2293d62caec94  run_claude.py
7da186a0a4be81735ad63484ef33c449  analyze.py
e919545680747028d08e199038663913  inputs/items_test.jsonl
dbb2fe818d712e837639d110eb47c820  inputs/games_test.json
9fced7decf3c8612717a59470b678f52  inputs/items_pilot.jsonl
```

## Amendments
(none yet)
