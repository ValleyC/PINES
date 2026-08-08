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

## Five-seed matched repair

The canonical repair aggregate is `repair_matched_v2_summary.json`. It covers
reset-to-value and floor-rounded fixed-point execution with five methods, five
seeds, identical 800-input calibration splits, and 280 updates for every
gradient method. Label-free methods use zero labels; source-initialized target
fine-tuning/QAT and random-initialized target retraining use all 800 labels.

Certificate-directed repair recovers 77.2% of reset loss and 87.8% of floor-
rounding loss. Every one of its ten cells clears 70% recovery and beats both
unrepaired deployment and global threshold scaling. It slightly beats logit-only
on reset but loses to it on floor rounding; labeled QAT recovers 101.5% on the
fixed-point condition. No method, seed, or condition certifies a five-point
budget, although none of the 50 bounds is violated.

The earlier `repair_reset_summary.json` is preserved but superseded for baseline
coverage. Regenerate the canonical aggregate from the ignored raw reports with:

```powershell
python experiments/aggregate_shd_repairs_matched.py
```

### Earlier reset-only aggregate

On reset-to-value, certificate-directed repair recovers 77.2% of lost test
accuracy on average, compared with 74.2% for pure logit distillation and 40.1%
for global threshold scaling. It clears 70% recovery for all five seeds.

This does not yield a deployable certificate. The mean post-repair upper bound
is 15.1 points, no seed is accepted at a five-point budget, and the bound is
1.85 points looser on average than logit-only distillation. Repair therefore
restores aggregate accuracy without restoring reference decision identity.

Regenerate the earlier repair aggregate from the ignored raw reports with:

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

A five-seed reference-member radius sweep explains why uniform refinement is not
a practical repair. Joint coverage averages 89.64% at relative radius 1e-6,
50.57% at 1e-5, and zero at 1e-4. Timestep-only and threshold-only sweeps show the
same cliff. At plus/minus 1%, an 8-by-8 grid still covers zero of 128 inputs, and
a 16-by-16 grid covers zero of 64 after 89 seconds. Reaching sub-box radius 1e-5
would require about 1,000 partitions per axis, or one million continuous boxes
before multiplying by 16 discrete semantics.

```powershell
python experiments/run_shd_continuous_family.py --seed 1701 --validation-samples 64
python experiments/aggregate_shd_continuous_family.py
python experiments/run_shd_continuous_radius_sweep.py --seed 1701
python experiments/aggregate_shd_continuous_radius_sweep.py
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

### Fixed-branch continuous diagnostic

The canonical endpoint diagnostic is `branch_stability_summary.json`. For the
reference discrete semantics, exact center-spike-trace identity across joint
timestep/threshold corners averages 99.28% at relative radius 1e-6, 93.98% at
1e-5, 54.12% at 1e-4, 0.74% at 1e-3, and zero at 1e-2. Prediction identity at
the same corners remains 89.36% at 1e-2.

Corner trace identity is necessary but not sufficient for a sound certificate
that assumes the center branch throughout the box. The result therefore rules
out that simple route at plus/minus 1%, while leaving useful headroom for a
fixed-branch analyzer at much smaller radii. A successful target-radius method
must represent correlated branch changes rather than merely avoid interval
merging along one trace.

```powershell
python experiments/run_shd_branch_stability.py --seed 1701
python experiments/aggregate_shd_branch_stability.py
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
