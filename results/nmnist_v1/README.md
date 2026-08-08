# N-MNIST feedforward negative control

This directory contains the tracked aggregate from the frozen five-seed
N-MNIST software matrix. The raw checkpoints, preprocessed events, and per-seed
semantic traces remain under the ignored `artifacts/` and `data/` directories.

Reproduce the aggregate from the raw artifacts with:

```text
python experiments/aggregate_nmnist.py
```

Key results across 50 seed-condition cells:

- Reference test accuracy: 97.416% mean, 0.115-point sample standard deviation.
- No target condition loses more than five points on average.
- No observed violation of the simultaneous 95% disagreement bound.
- Median bound slack: 1.15 points; 90th-percentile slack: 3.03 points.
- The frozen three-point median and eight-point 90th-percentile software gates
  both pass.
- Disagreement ranks absolute accuracy change with Pearson r = 0.986.

These are emulator results, not physical certificates. Together with the SHD
matrix, they show that the distribution-free bound can be tight for a robust
feedforward network while becoming operationally loose for a recurrent model
whose semantic changes create many compensating decision flips.
