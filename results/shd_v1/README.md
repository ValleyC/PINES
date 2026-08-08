# SHD v1 software evidence

This directory contains the compact, tracked aggregate of the frozen five-seed
software matrix. Large checkpoints, prediction arrays, and raw/processed data
remain under ignored `artifacts/` and `data/` directories; their hashes are
recorded in `software_matrix_summary_float32.json`.

The canonical aggregate is `software_matrix_summary_float32.json`. The original
v1 aggregate is preserved for audit but superseded for semantics metadata: its
runner executed float32 while its unquantized semantics object retained the
float64 default. The corrected artifact changes hashes and metadata only; all
predictions, accuracies, and bounds are identical.

## Result

- Reference accuracy: 71.46% mean, 2.95-point sample standard deviation.
- Five of ten target conditions lose more than five points on average.
- No observed violation among 50 simultaneous 95% bounds.
- Pearson disagreement/change correlation: 0.9893; Spearman: 0.9567.
- Median bound slack: 17.61 points; 90th percentile: 30.10 points.

The transport problem and label-free ranking signal are supported. The current
distribution-free accuracy-change certificate fails its preregistered tightness
gates and does not yet support the top-venue formulation. These results are
software-only and conditional on the emulator.

Regenerate with:

```powershell
python experiments/aggregate_shd.py
python experiments/correct_software_aggregate_float32.py --result-root results/shd_v1 --semantic-root artifacts/shd_v1_semantics_final --benchmark SHD
```

## Five-seed reset repair

On reset-to-value, certificate-directed repair recovers 77.2% of lost test
accuracy on average, compared with 74.2% for pure logit distillation and 40.1%
for global threshold scaling. It clears 70% recovery for all five seeds.

This does not yield a deployable certificate. The mean post-repair upper bound
is 15.1 points, no seed is accepted at a five-point budget, and the bound is
1.85 points looser on average than logit-only distillation. Repair therefore
restores aggregate accuracy without restoring reference decision identity.

Regenerate the repair aggregate from the ignored raw reports with:

```powershell
python experiments/aggregate_shd_repairs.py
```

## Static family certificate

The sound interval certifier was run on all 861 audit inputs for every seed.
Single-axis families certify 49.8% of inputs for reset, 76.0% for integration,
and 75.7% for timing or delay. Nearest fixed point certifies 95.0%.

The preregistered 16-member high-risk family certifies only 18.79% on average
(17.77–19.40% by seed), leaving 81.21% vacuous. Exact enumeration agrees across
the same family on 38.72%, so the mean interval-relaxation gap is 19.93 points.
No static certificate contradicts exact execution. Because every seed is
vacuous on more than 80% of inputs, this experiment triggers the plan's explicit
stop/pivot rule unless a substantially tighter sound abstraction reverses it.

Regenerate with:

```powershell
python experiments/run_shd_static_family.py --seed 1701
python experiments/aggregate_shd_static_family.py
```
