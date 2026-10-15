# Results: does Claude predict stolen-base success better than a model?

Pre-registration: `PREREG.md` (frozen in `48e69cf`, md5 `c040ba74`, no
amendments). The analysis is `analyze.py` exactly as frozen, run once on
2026-10-15 after all batches came back. Outcomes from 2026-07 on were not
looked at before that run. Every number below is in `published/results.json`
and `published/analyze.log`; the headline log losses were recomputed from the
raw outputs with separate code and agree to five digits.

## Verdict

**H1 and H2 are not supported.** Neither Claude on its own (B) nor Claude
given the model's probability (C) beats the model (A) on the 1,843 test
attempts (2026-07-01 to 09-27, 863 games).

| Arm | Log loss (test) | Brier | AUC |
|---|---|---|---|
| A: logistic model, league rate as offset | 0.46364 | 0.14535 | 0.593 |
| B: Claude, names and facts | 0.46685 | 0.14620 | 0.578 |
| C: Claude, names, facts and A's probability | 0.46239 | 0.14486 | 0.598 |
| L: league rate (reference) | 0.47132 | 0.14751 | 0.482 |

| Difference (test) | Mean | 95% interval | Holm-adjusted p |
|---|---|---|---|
| C − A (H1) | −0.0012 | −0.0038 to +0.0013 | 0.317 |
| B − A (H2) | +0.0032 | −0.0032 to +0.0098 | 0.835 |

As the pre-registration says, the test can only detect differences of about
0.010 in log loss. Both intervals stay inside ±0.010, so the reading is "no
difference of that size": Claude did not add or lose as much as 0.010 against
the model.

## Memory check and August–September

Claude's training data reaches mid-2026, so it may know how the first half
went. The rule compares D = (A − arm) on the April–June control with D on the
test set.

- **The check does not fire for either arm.** D_control − D_test is −0.033 to
  −0.003 for B and −0.013 to +0.0002 for C. Neither interval lies above 0,
  and B's lies entirely below it, the opposite of what memorization would
  produce.
- On the control (300 attempts, April–June), **B is clearly worse than the
  model**: B − A = +0.021 (95% interval +0.008 to +0.035). C − A = +0.005
  (−0.001 to +0.011).
- On August–September only (1,238 attempts): B − A = +0.005 (−0.003 to
  +0.013), C − A = −0.001 (−0.004 to +0.002).

## Other numbers reported regardless of outcome

- **Calibration**: the test success rate is 0.820. A predicts 0.810 on
  average, B 0.795. B places 743 of 1,843 attempts in the 0.7 bin, where 79%
  succeeded against a predicted 76%.
- **How much C moves A**: the mean |C − A| is 0.0175. The correlation between
  C's adjustment (C − A) and the residual (y − A) is +0.063.
- **Runners with fewer than 5 earlier attempts** (190): A 0.4816, B 0.4881,
  C 0.4792.
- **S1, missing sprint speed or pop time** (723 test attempts): B's gain over A
  is −0.0008 on those rows against −0.0048 where both are present (one-sided
  p 0.279, not supported). AUC on the missing rows: A 0.585, B 0.572.
- **Asking B twice** (100 test attempts): the two answers differ by 0.014 on
  average (sd of the difference 0.024).
- **Model and cost**: every answer came from `claude-opus-5-5` at effort
  "medium", with no request failing or needing a resubmission. B used 2.49M
  input and 0.36M output tokens (batch price USD 8.61), C used 2.70M and 0.36M
  (USD 9.03).

## What this does and does not say

- It concerns predicting whether an attempt succeeds. It says nothing about
  whether a steal was the right call, because teams choose which attempts to
  make.
- B tends to be less sure of success than the outcomes justify, and on its own
  it ranks attempts worse than the model (AUC 0.578 against 0.593). Given the
  model's number, C stays close to it, and the remaining adjustments carry
  little information (correlation +0.063 with the residual).

## Files

- `published/results.json`: everything `analyze.py` writes.
- `published/analyze.log`: its printed summary.
- `published/raw/*.jsonl.gz`: every Claude answer (B and C on test and control,
  the 100 repeats and the 40-call pilot), with model, stop reason and token
  usage.
