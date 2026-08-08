# N-MNIST feedforward negative control

This directory contains the tracked aggregate from the frozen five-seed
N-MNIST software matrix. The raw checkpoints, preprocessed events, and per-seed
semantic traces remain under the ignored `artifacts/` and `data/` directories.

The canonical aggregate is `software_matrix_summary_float32.json`. The original
v1 aggregate is preserved for audit but superseded for semantics metadata: its
runner executed float32 while its unquantized semantics object retained the
float64 default. The corrected artifact changes hashes and metadata only; all
predictions, accuracies, and bounds are identical.

Reproduce the aggregate from the raw artifacts with:

```text
python experiments/aggregate_nmnist.py
python experiments/correct_software_aggregate_float32.py --result-root results/nmnist_v1 --semantic-root artifacts/nmnist_v1_semantics --benchmark N-MNIST
python experiments/run_nmnist_static_family.py --seed 1701 --max-samples 128
python experiments/aggregate_nmnist_static_family.py
```

Key results across 50 seed-condition cells:

- Reference test accuracy: 97.416% mean, 0.115-point sample standard deviation.
- No target condition loses more than five points on average.
- No observed violation of the simultaneous 95% disagreement bound.
- Median bound slack: 1.15 points; 90th-percentile slack: 3.03 points.
- The frozen three-point median and eight-point 90th-percentile software gates
  both pass.
- Disagreement ranks absolute accuracy change with Pearson r = 0.986.
- On deterministic 128-input subsets of each frozen audit split, the sound
  interval certificate covers 96.72% of inputs for the full 16-member semantic
  family. Exact family agreement is 98.59%, leaving only a 1.88-point relaxation
  gap. This diagnostic subset is not a replacement for the full audit.

These are emulator results, not physical certificates. Together with the SHD
matrix, they show that the distribution-free bound can be tight for a robust
feedforward network while becoming operationally loose for a recurrent model
whose semantic changes create many compensating decision flips.
