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

The canonical finite-family aggregate is `static_family_memberwise_summary.json`.
It checks the argmax within each mutually exclusive discrete semantics member
before taking the family conjunction. Across all 861 audit inputs per seed,
single-axis coverage is 76.19% for reset, 84.81% for integration, and 83.00% for
timing or delay. The 16-member family covers 38.72% and exactly matches enumerated
agreement for unquantized point semantics, so every seed passes the 20% gate.

The preserved `static_family_summary.json` is sound but superseded for this
claim. It merged logits across mutually exclusive semantics and covered only
18.79%, creating an avoidable 19.93-point cross-member relaxation gap.

Regenerate with:

```powershell
python experiments/run_shd_static_family.py --seed 1701 --output-root artifacts/shd_v2_memberwise_static_family
python experiments/aggregate_shd_memberwise_static_family.py
```

### Continuous bounded family

Joint timestep and threshold uncertainty of only plus/minus 1%, 2%, or 5% is
fully vacuous under the current sound interval domain: every seed certifies zero
of 861 SHD inputs at every radius. Endpoint-grid agreement on fixed 64-input
subsets remains 36.9%, 32.5%, and 24.7%, respectively, with no counterexample to
a static certificate. Endpoint agreement is diagnostic, not a proof over the
continuous interior. This continuous result—not finite enumeration—triggers the
declared pivot gate.

```powershell
python experiments/run_shd_continuous_family.py --seed 1701 --validation-samples 64
python experiments/aggregate_shd_continuous_family.py
```

### Recurrence intervention

Zeroing only the trained recurrent matrix, while retaining the same SHD inputs
and every other parameter, raises both finite-family static coverage and exact
agreement from 38.72% to 89.31%, with no observed soundness failure. This
establishes recurrent feedback as a major mechanism behind true semantic
instability, but it is not an accuracy-matched comparison and does not repair the
continuous-box abstraction.

Regenerate with:

```powershell
python experiments/run_shd_static_family.py --seed 1701 --output-root artifacts/shd_v2_memberwise_zero_recurrence --families full --zero-recurrence
python experiments/aggregate_shd_recurrence_ablation.py --input-root artifacts/shd_v2_memberwise_zero_recurrence --suffix memberwise --figure-stem recurrence_ablation_memberwise --trained-summary results/shd_v1/static_family_memberwise_summary.json --feedforward-summary results/nmnist_v1/static_family_memberwise_summary.json
```

### Horizon diagnostic

On deterministic 256-input audit subsets, trained recurrent finite-family static
coverage rises from 30.94% at 10 bins to 36.17% at 50 bins and tracks exact
agreement. The zero-recurrence trajectory rises from 66.72% to 88.98%. Agreement
does not monotonically decay with time because reference margins accumulate.
Shortening the unroll therefore does not address the continuous-box failure.

Regenerate with:

```powershell
python experiments/run_shd_horizon_diagnostic.py --seed 1701 --max-samples 256 --output-root artifacts/shd_v2_memberwise_horizon
python experiments/aggregate_shd_horizon_diagnostic.py --input-root artifacts/shd_v2_memberwise_horizon --suffix memberwise_v2 --figure-stem shd_horizon_memberwise_v2
```
