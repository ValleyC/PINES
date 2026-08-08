# SHD v1 software evidence

This directory contains the compact, tracked aggregate of the frozen five-seed
software matrix. Large checkpoints, prediction arrays, and raw/processed data
remain under ignored `artifacts/` and `data/` directories; their hashes are
recorded in `software_matrix_summary.json`.

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
