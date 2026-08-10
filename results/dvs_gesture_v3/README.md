# DVS Gesture v3 development evidence

This directory contains the tracked aggregate of the five-seed convolutional/
recurrent DVS Gesture software matrix. Large checkpoints, predictions, raw
archives, and processed event windows remain under ignored `artifacts/` and
`data/` directories; their hashes are recorded in the summary.

Protocol:

- Canonical IBM/Figshare archives verified against Tonic's published MD5s.
- Four fixed 1.5-second windows per recording at 25 ms per semantic step.
- Splits are hash-partitioned by recording identity, so windows never cross
  training, label-free repair, or label-free audit partitions.
- Five fixed seeds use the same 32/64-channel spiking convolutional front end
  and 256-neuron recurrent spiking layer.

Key results across 50 seed-condition cells:

- Reference test accuracy: 84.85% mean, 1.76-point sample standard deviation.
- Floor-rounded fixed point loses 36.59 points on average.
- No observed violation of the simultaneous 95% disagreement bound.
- Disagreement ranks absolute change with Pearson r = 0.962.
- Median bound slack: 18.54 points; 90th-percentile slack: 29.42 points.
- Only two conditions lose more than five points on average.

These results support the transport-sensitivity and label-free ranking claims,
but fail both certificate-tightness gates. The audit has only 104 recordings per
seed, which creates a roughly five-point simultaneous-confidence floor even
near zero observed disagreement.

This is development evidence, not a final sequestered result: official test
accuracy was used to select the v3 reference pipeline before any semantic target
was executed. A fresh, externally sequestered replication is required for a
submission claim.

Regenerate the aggregate with:

```powershell
python experiments/aggregate_dvs_gesture.py
```

## Fixed-point repair development study

The canonical repair aggregate is `repair_floor_task_tuned_clean_v5_summary.json`.
Certificate-directed and logit-only repair use the same 97 unlabeled calibration
recordings, 960 trainable scale/bias parameters, and 280 updates. QAT receives
all 97 labels, exposes 678,560 parameters, and uses the same update count. Full
target retraining uses the same 97 labels and 280 updates from a random
initialization. It is therefore a matched-label from-scratch baseline rather
than retraining on the full original training set.

The frozen DVS development objective uses margin and logit weights of 1.0 and
1.0. This task-level choice was made before the required fresh sequestered
confirmation.

- Task-tuned certificate-directed recovery: 94.9% mean, 5.7-point standard deviation.
- Logit-only recovery: 89.3% mean, 8.4-point standard deviation.
- Global-threshold recovery: 74.8% mean, 8.6-point standard deviation.
- Labeled QAT recovery: 99.6% mean, 10.8-point standard deviation.
- Supervised from-scratch recovery: 6.0% mean, 29.0-point standard deviation.
- Mean simultaneous bounds: 17.4, 15.1, 33.6, 22.3, and 59.9 points,
  respectively.

The proposed method clears 70% recovery for every seed, beats logit-only in four
of five paired seeds, and improves mean recovery by 5.6 points. Its mean bound
is nevertheless 2.3 points looser, and no method certifies a five-point budget.
This supports a cross-architecture label-free repair advantage on recovery, not
universal metric dominance or restoration of a tight certificate. It remains
development evidence because the reference pipeline was selected using official
test accuracy before semantic evaluation.

Regenerate the superseding repair aggregate with:

```powershell
python experiments/aggregate_dvs_repairs.py --certificate-root artifacts/dvs_gesture_v9_balanced_clean_da9c0f9_v1 --logit-root artifacts/dvs_gesture_v4_repairs --global-root artifacts/dvs_gesture_v4_repairs --include-qat --qat-root artifacts/dvs_gesture_v5_qat_nll --include-supervised --supervised-root artifacts/dvs_gesture_v7_supervised_retraining --suffix task_tuned_clean_v5
```

The earlier `repair_floor_summary.json` and `repair_floor_matched_v2_summary.json`
are preserved; v2 used a logits-style surrogate on already aggregated
probabilities and is superseded only for the QAT comparison.
