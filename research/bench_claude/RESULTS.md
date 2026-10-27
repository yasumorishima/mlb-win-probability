# Results — Claude's bench calls vs the managers'

Pre-registration: `PREREG.md` (frozen `78a0a97`, merged `7a98f7a`; Amendment 1 merged `3c544bd` before any test answer was read). Run once, in Claude Code cloud, 2026-10-27,
model(s) claude-opus-5-5, live Messages API calls (Amendment 1: the batches stalled and were cancelled unread). Cost from the recorded tokens at full price: B $7.42, C $7.56 (analyze.py's `cost_usd_batch` is the batch rate, half of this). Also billed: the pilot (46 live calls) and the requests the cancelled batches had finished (B 307, C 293; about $4 at the batch rate; never read).

Agreement = the arm picked the player the manager actually sent in. It measures how much the picks look like real MLB managers' choices, not whether they were right.

## Hypotheses (pitching changes)

- **H1 C > MODEL (pitching changes)**: +10.4 pt (95% +6.9 to +14.1), Holm p = 0.000 → **SUPPORTED**
- **H2 C > USAGE (pitching changes)**: +1.9 pt (95% -3.6 to +7.6), Holm p = 0.545 → **NOT SUPPORTED**

## Pitching changes (6th inning on)

422 items in 79 games; mean 7.3 candidates (chance 14.9%). Invalid answers: B 1, C 0.

| arm | agreement | top-3 agreement |
|---|---|---|
| B | 32.7% | 73.5% |
| C | 30.6% | 65.9% |
| MODEL | 20.1% | 51.9% |
| USAGE | 28.7% | — |

Paired differences (bootstrap over games):

- C-MODEL: +10.4 pt (95% +6.9 to +14.1)
- B-MODEL: +12.6 pt (95% +7.7 to +17.5)
- C-B: -2.1 pt (95% -6.3 to +2.0)
- C-USAGE: +1.9 pt (95% -3.6 to +7.6)
- B-USAGE: +4.0 pt (95% -1.2 to +9.3)

| innings | n | B | C | MODEL | USAGE |
|---|---|---|---|---|---|
| 6-7 | 180 | 24.4% | 21.1% | 12.2% | 17.8% |
| 8+ | 242 | 38.8% | 37.6% | 26.0% | 36.8% |

## Pinch hitters

165 items in 70 games; mean 3.7 candidates (chance 28.8%). Invalid answers: B 0, C 0.

| arm | agreement | top-3 agreement |
|---|---|---|
| B | 39.4% | 90.9% |
| C | 40.0% | 89.1% |
| MODEL | 38.2% | 88.5% |

Paired differences (bootstrap over games):

- C-MODEL: +1.8 pt (95% -2.6 to +6.5)
- B-MODEL: +1.2 pt (95% -6.5 to +8.5)
- C-B: +0.6 pt (95% -5.6 to +7.0)

Excluded items: {'actual_not_in_candidates': 30, 'fewer_than_2_candidates': 3}.
