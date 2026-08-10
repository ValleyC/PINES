# Repair diagnosis and corrective experiments, 2026-08-09

## Bottom line

The weak repair result had two different causes.  One was an implementation
defect.  The other was an objective imbalance.  Both are now corrected in the
development code and evaluated on the untouched repair-audit splits.

The corrected task-tuned objective improves the mean PINES repair result from
72.1/14.8, 89.7/18.5, and 91.1/21.5 to 80.6/12.7, 91.0/16.8, and 94.9/17.4,
where each pair is lost accuracy recovered in percent/post-repair simultaneous
upper bound in points for SHD reset, SHD floor-fixed-point, and DVS Gesture
floor-fixed-point.  Every seed now clears 70% recovery in all three conditions.

This is a real improvement, but it does not restore a five-point certificate.
The correct claim remains label-free transport repair followed by mandatory
recertification.  It is not certificate restoration.

## Implementation defect

The SHD repair optimizer previously executed all operations in float32, then
exported and audited the repaired model with the declared operational semantics:
binary64 arithmetic with explicit float32 storage casts.  The hard forward pass
used for optimization could therefore differ from the executor used for model
selection and certification near a threshold.

The repairable and supervised SHD modules now:

- retain binary64 working arithmetic,
- apply declared float32 casts with a straight-through gradient,
- execute fixed-point rounding and saturation exactly in the hard forward pass,
- cast input events to the executor working type, and
- match the scalar/vectorized operational executor exactly on state, spike,
  logit-trace, and final-logit tests for float32, reset-to-value, and floor-fixed
  semantics.

This correction is required for validity even where it does not improve a
particular accuracy cell. The final cleaned suite has 132 passing tests. The
rejected held-out-selection experiment and its dedicated test were removed
after the larger development check failed.

## Objective diagnosis

The former PINES loss weighted the source-logit term by 0.05 while assigning
unit weight to the hard source-prediction margin term.  It fit the unlabeled
calibration predictions aggressively but generalized poorly:

| Condition | Method | Calibration disagreement | Audit disagreement | Gap |
|---|---:|---:|---:|---:|
| SHD reset | former PINES | 5.80% | 12.31% | 6.51 points |
| SHD reset | logit-only | 10.75% | 11.59% | 0.84 points |
| SHD fixed | former PINES | 9.10% | 16.75% | 7.65 points |
| SHD fixed | logit-only | 14.88% | 15.54% | 0.67 points |

The low calibration count was therefore not evidence of a better deployment
model. It was a checkpoint-selection overfitting signature.

The adopted development correction keeps the margin, spike, state, logit, and
parameter regularization terms, but raises the normalized source-logit weight
from 0.05 to 1.0. A frozen margin-weight sweep then selects 0.25 for SHD and
retains 1.0 for DVS Gesture. This preserves the certificate-directed component
while using source output geometry as a strong regularizer. The task-level
weights were selected on development evidence and require a fresh sequestered
audit before they can support a confirmatory claim.

## Five-seed matched results

All methods use the same model, target semantics, 800 SHD or 97 DVS unlabeled
calibration inputs, 40 epochs, parameterization, and untouched audit split.
DVS Gesture remains development evidence because its reference pipeline was
selected before the semantic study was frozen.

| Task and target | PINES recovery | PINES bound | Logit recovery | Logit bound |
|---|---:|---:|---:|---:|
| SHD reset-to-value | 80.6% | 12.7 | 70.5% | 13.5 |
| SHD floor fixed point | 91.0% | 16.8 | 89.7% | 17.7 |
| DVS floor fixed point | 94.9% | 17.4 | 89.3% | 15.1 |

Task-tuned PINES improves mean recovery over logit-only in all three conditions.
It also tightens the mean bound by 0.84 points on SHD reset and 0.88 points on
SHD fixed point. Its DVS bound is 2.30 points looser. These differences must
remain visible in the manuscript. The result supports a stronger recovery claim
and a cross-metric advantage on SHD, not universal dominance on every task.

The largest stability improvement is SHD seed 8119 under reset. Recovery rises
from 50.0% with the corrected former loss to 86.2%, while the bound falls from
14.31 to 11.46 points. All ten SHD seed-condition cells now recover at least
70% of the lost accuracy.

## Rejected alternatives

Two targeted alternatives were tested and are not adopted:

1. A source-margin-deficit loss reduced the mean reset bound slightly but did
   not improve mean recovery and worsened the fixed-point bound.
2. A held-out 20% selection subset reduced the data available for optimization.
   On the development seed it worsened both repair columns, even after matching
   the number of gradient updates.

These are kill tests.  Their favorable individual cells must not be selected
post hoc for the paper.

## What code cannot fix

The remaining large family bounds are not primarily repair-code defects.

- Exact enumeration finds genuine prediction changes on 61.3% of SHD and
  42.3% of DVS inputs for the deliberately broad 16-member semantics family.
  No sound analyzer can certify those inputs as invariant.
- The continuous SHD reset audit leaves 29.3% unresolved at the frozen proof
  budget, so tighter relational analysis could reduce conservatism.  However,
  independent searches already falsify 36.8% of inputs.  The broad contract
  would still reject a five-point budget even with a perfect analyzer.
- The DVS certificate audit contains only 104 independent recordings.  With
  ten-way simultaneous confidence correction, the statistical floor is large.
  More independent recordings, not more optimization steps, are required for
  a five-point certificate with nonzero disagreements.

## Evidence and remaining checks

Tracked aggregate diagnostics are:

- `results/shd_v1/repair_task_tuned_clean_v5_summary.json`
- `results/dvs_gesture_v3/repair_floor_task_tuned_clean_v5_summary.json`
- `results/shd_v1/repair_task_tuned_objective_diagnostic_v3.json`
- `results/dvs_gesture_v3/repair_task_tuned_objective_diagnostic_v3.json`
- `results/shd_v1/repair_task_tuned_family_grid_v5.json`

With the ignored raw artifacts present, the complete hash, revision, split, and
gate audit is reproduced by `python experiments/verify_repair_evidence.py`.

The corrected-executor QAT and supervised baselines are complete. The sampled
post-repair family diagnostic is also complete over all 861 SHD audit inputs
and five seeds. At a 9 by 9 grid over the plus or minus 1% contract, PINES
raises reset-family identity over corrected logit-only by 0.26 points on average
with mixed per-seed signs. For floor fixed point it improves identity by 1.37
points and every seed is positive. These are finite-grid diagnostics and not
continuous-family certificates.

Before the current development numbers are treated as final submission
evidence:

1. freeze a fresh sequestered audit because the present SHD audit has been used
   during method development,
2. repeat the DVS study with a reference pipeline selected without target-test
   feedback, and
3. obtain the physical-backend conformance evidence.
