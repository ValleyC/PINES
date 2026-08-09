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

The canonical repair aggregate is
`repair_matched_cast_faithful_v4_summary.json`. It covers reset-to-value and
floor-rounded fixed-point execution with five methods, five seeds, identical
800-input calibration splits, and 280 updates for every gradient method. The
target executor performs the declared float32 state casts. Label-free methods
use zero labels; source-initialized target fine-tuning/QAT and random-initialized
target retraining use all 800 labels.

Certificate-directed repair recovers 72.1% of reset loss and 89.7% of floor-
rounding loss. Nine of its ten cells clear 70% recovery; reset seed 8119 reaches
only 23.1%. It beats both unrepaired deployment and global threshold scaling in
every cell, but logit-only has higher mean recovery in both conditions. Labeled
QAT recovers 101.5% on the fixed-point condition. No method, seed, or condition
certifies a five-point budget, although none of the 50 bounds is violated. The
older `repair_matched_v2_summary.json` used the neighboring full-float32
executor and is retained only as superseded development provenance.

The earlier `repair_reset_summary.json` is preserved but superseded for baseline
coverage. Regenerate the canonical aggregate from the ignored raw reports with:

```powershell
python experiments/aggregate_shd_repairs_matched.py --artifact-root artifacts/shd_v75_repairs_cast_faithful_v1 --result-stem repair_matched_cast_faithful_v4 --figure-stem shd_repair_matched_cast_faithful_v4
```

### Earlier reset-only aggregate

Under the superseded full-float32 executor, certificate-directed repair recovered
77.2% of reset loss on average, compared with 74.2% for pure logit distillation
and 40.1% for global threshold scaling, and cleared 70% for all five seeds. The
cast-faithful aggregate above replaces these numbers in current claims.

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

### Branch complexity and full-family feasibility ceiling

The canonical finite-grid diagnostics are `branch_complexity_summary.json` and
`full_family_grid_summary.json`. At a 17-by-17 reference-member grid, the mean
input has 81.67 sampled spike traces, while 80.94% of inputs retain a single
prediction. A 9-by-9 grid across all 16 discrete members retains 31.09% full-
family prediction identity (28.12% to 35.16% by seed). The latter is an empirical
upper bound on achievable unchanged-argmax certificate coverage and remains
above the 20% gate, but only narrowly. Neither finite grid is a proof over the
continuous interior.

```powershell
python experiments/run_shd_branch_complexity.py --seed 1701
python experiments/aggregate_shd_branch_complexity.py
python experiments/run_shd_full_family_grid.py --seed 1701
python experiments/aggregate_shd_full_family_grid.py
```

`subfamily_grid_summary.json` decomposes the same 9-by-9 grid into targeted
contracts. Reset retains 60.31% sampled identity, integration 68.91%, timing and
delay 69.06%, reset plus delay 46.72%, integration plus timing 55.47%, and the
full family 31.09%. This supports targeted bounded-axis contracts as the fallback
formulation; all values remain empirical ceilings rather than proofs.

```powershell
python experiments/run_shd_subfamily_grid.py --seed 1701
python experiments/aggregate_shd_subfamily_grid.py
```

### Explicit branch-set kill diagnostic

`branch_set_feasibility_summary.json` records a sound, single-seed feasibility
test of explicit spike-vector separation. No staged full-cover input completes.
At a 65,536-state cap, neither merged nor unmerged analysis completes any tested
representative cell at 32, 64, or 128 partitions per axis. Finer cells delay the
median abort from timestep 21 to 41 but imply 16,384 cells per input at the
finest setting. Cap exhaustion is inconclusive, never a certificate failure.
The result rejects explicit branch lists as a practical full-model solution and
points to symbolic guard compression or decision-level correlated bounds.

```powershell
python experiments/run_shd_branch_set_partition.py --seed 1701
python experiments/run_shd_branch_set_cell_sweep.py --seed 1701 --partitions 32 64 128 --max-branches 65536
python experiments/run_shd_branch_set_cell_sweep.py --seed 1701 --partitions 32 64 128 --max-branches 65536 --merge
python experiments/aggregate_shd_branch_set_feasibility.py
```

### Decision-margin interval kill diagnostic

`decision_margin_domain_summary.json` records a sound output-correlated domain.
It propagates every reference-versus-competitor margin directly, preventing one
uncertain spike from independently worsening both sides of a comparison. A unit
test proves strict improvement over separate logit intervals on a constructed
witness, and sampled executions are enclosed. Nevertheless, staged SHD reset-
family audits certify zero inputs through 32-by-32 partitioning, exactly matching
ordinary intervals. The unresolved loss is in recurrent state and guard
correlation, not at the output comparison.

```powershell
python experiments/run_shd_decision_margin_certificate.py --seed 1701 --partitions 1 2 4 --sample-count 128
python experiments/run_shd_decision_margin_certificate.py --seed 1701 --partitions 8 16 --sample-count 32 --output-root artifacts/shd_v23_decision_margin_fine
python experiments/run_shd_decision_margin_certificate.py --seed 1701 --partitions 32 --sample-count 16 --output-root artifacts/shd_v24_decision_margin_p32
python experiments/aggregate_shd_decision_margin_domain.py
```

### Adaptive margin-cover kill diagnostic

`adaptive_margin_domain_summary.json` asks whether nonuniform subdivision can
focus work on difficult guard regions. It first selects inputs stable at all 81
points of a finite 9-by-9 grid; 80 of the first 128 frozen audit-pool inputs meet
that diagnostic. For the first two, no sub-box is retired through 256 adaptive
leaves. The first input still has 0% certified parameter volume after 8,191 box
analyses and 4,096 final leaves. Full certification is never claimed from finite
grid stability or partial volume.

```powershell
python experiments/run_shd_adaptive_margin_certificate.py --seed 1701 --max-leaves 64 256
python experiments/run_shd_adaptive_margin_certificate.py --seed 1701 --sample-count 1 --max-leaves 1024 4096 --output-root artifacts/shd_v26_adaptive_margin_fine
python experiments/aggregate_shd_adaptive_margin_domain.py
```

### Relational affine-guard advance

`affine_guard_domain_summary.json` records the first sound continuous-domain
method with nonzero local SHD coverage. It carries shared timestep and threshold
generators through recurrent states and introduces residual intervals only for
nonlinear products and threshold-uncertain spikes. Uniform local cell-input
coverage grows from 0.15% at 4-by-4 to 3.74% at 8-by-8 and 18.46% at 16-by-16;
the output-correlated interval baseline remains at zero.

On the first finite-grid-stable input, adaptive affine certified parameter
volume grows from 3.13% at 64 leaves to 30.66%, 61.47%, 85.72%, and 95.61% at
256, 1,024, 4,096, and 16,384 leaves. This is not a full certificate: unresolved
guard-surface cells remain, and certified volume is not a probability over
semantics. The route advances to symbolic guard-surface cuts, not to five seeds.

`guard_guided_affine_summary.json` compares width-based bisection with an affine
guard-sensitivity axis policy. The guard policy improves coverage by 1.37 points
at 1,024 leaves and 0.63 points at 4,096, but is slightly worse at 256. This
small, non-monotone gain ends axis-policy tuning; it does not alter the missing
full-certificate verdict.

```powershell
python experiments/run_shd_affine_guard_certificate.py --seed 1701 --partitions 1 2 4 8
python experiments/run_shd_affine_guard_certificate.py --seed 1701 --partitions 16 --sample-count 32 --output-root artifacts/shd_v28_affine_guard_p16
python experiments/run_shd_adaptive_margin_certificate.py --seed 1701 --domain affine_guard --sample-count 1 --max-leaves 64 256 1024 4096 --output-root artifacts/shd_v29_adaptive_affine
python experiments/run_shd_adaptive_margin_certificate.py --seed 1701 --domain affine_guard --sample-count 1 --max-leaves 16384 --output-root artifacts/shd_v30_adaptive_affine_p16384
python experiments/aggregate_shd_affine_guard_domain.py
python experiments/run_shd_adaptive_margin_certificate.py --seed 1701 --domain affine_guard --split-strategy guard --sample-count 1 --max-leaves 64 256 1024 4096 --output-root artifacts/shd_v31_guard_guided_affine
python experiments/aggregate_shd_guard_guided_affine.py
```

### Polygonal recurrent guard cuts

`polygonal_guard_rounding_summary.json` is the canonical result. Convex
half-space clipping follows recurrent guard surfaces before width-scaled axis
refinement. Every float32 conversion of current, state, output contribution, and
logit receives a conservative rounding envelope; output errors are charged to
direct pairwise margins so shared spike terms still cancel. The older
`polygonal_guard_cut_summary.json` omitted this explicit arithmetic term and is
retained only as a superseded arithmetic-idealized development artifact.

With rounding included, a fixed 128-cut cap covers 82.26% at 1,024 leaves and
94.13% at 4,096. Under the scale-invariant one-cut-per-eight-leaves rule,
coverage reaches 99.7557% at 4,096 leaves and 99.8331% at 16,384. The largest
run analyzes 31,060 polygons and leaves 13,158 tiny polygons covering 0.1669%
of the original rectangle unresolved. Moving from 4,096 to 16,384 leaves gains
only 0.077 percentage points, so further leaf scaling is stopped.

This is diagnostic parameter area, not a probability over semantics. By itself,
the cover is inconclusive because a per-input family certificate requires every
positive-area region to be proved. The residual was subsequently closed by the
sound local branch oracle described below; the polygon-only artifact remains an
auditable intermediate result.

```powershell
python experiments/run_shd_guard_cut_certificate.py --seed 1701 --max-leaves 1024 4096 --max-guard-band-splits 128 --output-root artifacts/shd_v47_rounding_guard_cap128
python experiments/run_shd_guard_cut_certificate.py --seed 1701 --max-leaves 4096 --max-guard-band-splits 512 --output-root artifacts/shd_v48_rounding_guard_cap512
python experiments/run_shd_guard_cut_certificate.py --seed 1701 --max-leaves 16384 --max-guard-band-splits 2048 --output-root artifacts/shd_v49_rounding_guard_cap2048
python experiments/aggregate_shd_polygonal_guard_rounding.py
```

### Exact threshold-boundary slices

`exact_threshold_slices_summary.json` records an exact reset-to-value subproblem.
At a fixed timestep, threshold affects spike choices but not the conditional
state update, so the oracle partitions the requested interval into contiguous
spike-trace cells. It covers every representable binary64 threshold scale using
the emulator's actual multiplication and comparison, rather than sampled points
or relaxed guards.

For the same finite-grid-stable SHD input, all 257 fixed timestep slices across
plus/minus 1% preserve the reference class. The run exhausts 2,716 exact cells
in 12.1 seconds; individual slices need 6--17 cells. This rules out
threshold-only boundary relaxation at those slices. It is not a joint
continuous certificate because the open timestep regions between slices remain
unproved. More timestep-grid densification was stopped; the later local branch
oracle closes the joint residual directly.

```powershell
python experiments/run_shd_exact_threshold_slices.py --seed 1701 --output-root artifacts/shd_v45_exact_threshold_slices
python experiments/aggregate_shd_exact_threshold_slices.py
```

### Residual polygon geometry

`guard_residual_geometry_summary.json` characterizes the exact input passed to
the later local branch solver. The corrected 4,096-leaf run exports 3,365 unresolved convex
polygons to a hashed 111-KB archive. Their median actual timestep-factor width is
4.19e-5, below the 7.8125e-5 spacing of the 257 exact slices, and 1,271 polygons
fall entirely between slice locations. Although individually small, their
projections collectively span both complete parameter axes. This rejects more
uniform slice densification and motivated reasoning within the polygon
constraints.

```powershell
python experiments/run_shd_guard_cut_certificate.py --seed 1701 --max-leaves 4096 --max-guard-band-splits 512 --save-residual-polygons --output-root artifacts/shd_v50_rounding_guard_residuals
python experiments/aggregate_shd_guard_residual_geometry.py
```

### Residual-targeted counterexample search

`residual_counterexample_search_summary.json` records a falsification attempt
inside the exact unresolved geometry. A validated GPU parameter-batch executor
evaluates every polygon vertex, edge midpoint, and centroid plus 32 deterministic
random convex combinations per polygon. Across 128,460 unique parameter points,
every execution retains class 2 and the minimum reference-class margin is 1.033.

This is strong evidence that the residual reflects abstraction slack, but it is
not a certificate and does not cover unsampled points. More random search was
stopped in favor of the sound local polygon branch oracle.

```powershell
python experiments/run_shd_residual_counterexample_search.py --seed 1701 --random-points-per-polygon 32 --output-root artifacts/shd_v52_residual_counterexample_search_dense
python experiments/aggregate_shd_residual_counterexample_search.py
```

### Sound local branch closure and frozen screen

`polygon_branch_certificate_summary.json` records the first complete per-input
joint continuous-family certificate. The local oracle enumerates both outcomes
of every guard that remains uncertain inside a residual polygon, deliberately
retaining infeasible branches. All 3,365 residual polygons complete; every
retained trace predicts class 2, with median peak branch count 2, 99th
percentile 3, and maximum 14. Combining this closure with the 99.7557% affine
cover reaches total area one within 6.8e-12. The input was used during method
development, so this result is not population evidence.

The method budgets were then frozen at 64 polygon leaves, 64 branches, and eight
guard cuts. `hybrid_family_audit_summary.json` aggregates the first 32 untouched
certificate-audit inputs from each of five seeds. No labels, predictions,
margins, or grid behavior were used for selection. The full joint reset-to-value
box is certified for 58/160 model--input evaluations (36.25%), with
31.25--43.75% coverage by seed. The same 32 input positions recur across models,
so no pooled binomial interval is reported. Mean certified parameter
volume is 63.78%, and the area cover closes on every input. The frozen 20%
advance gate passes.

An independent 9-by-9 grid over the same continuous box finds no mismatch for
any certified input. It is stable on 112/160 inputs and finds concrete
prediction changes on 48; all 48 are left uncertified. The remaining 54
grid-stable but inconclusive inputs quantify proof- or budget-conservatism. This
grid is a falsification diagnostic, not part of the proof. The screen triggered
the separately frozen all-input audit over 4,305 certificate-split inputs.

`hybrid_family_full_audit_summary.json` is retained as a superseded development
artifact. Its 1,398/4,145 pooled headline and grid/Sobol counts must not be used
as current manuscript evidence. The soundness-corrected v68 audit keeps the same
domain, inputs, and 64/64/8 scientific budgets while adding reduction-error and
polygon-cover safeguards. Seed 1701 is complete at 347/861 certificates (40.30%)
and 61.33% mean proved volume; three prior certificates are withdrawn and two are
newly proved. The other seeds and canonical falsification are in progress. Final
coverage will report input-cluster and training-seed uncertainty separately.

```powershell
python experiments/run_shd_residual_polygon_branch_certificate.py --seed 1701 --output-root artifacts/shd_v56_residual_polygon_branches_all
python experiments/aggregate_shd_polygon_branch_certificate.py
python experiments/run_shd_hybrid_family_audit.py
python experiments/validate_shd_hybrid_audit_grid.py
python experiments/aggregate_shd_hybrid_audit.py
python experiments/run_shd_hybrid_family_full_audit.py --config configs/experiments/shd_hybrid_full_audit_v3.json --output-root artifacts/shd_v68_hybrid_full_audit_soundness_corrected_v1
```

### Selected-input 16-member continuous Cartesian certificate

`cartesian_member_scaling_sound_v5_summary.json` records a complete sound
certificate over all 16 combinations of integration rule, threshold timing,
reset rule, and zero/one-step synaptic delay for one label-free selected SHD
input. Every member also contains the complete plus/minus 1% continuous
timestep--threshold box. The canonical selector reproduces 236/800 stable
repair-calibration inputs (29.5%) and selects dataset index 862 by minimum
reference-class margin without labels.

Each mutually exclusive member receives its own frozen leaf budget before the
member results are conjoined. Mean proved area is 15.47%, 70.97%, and 100% at
16, 64, and 256 leaves per member; zero, four, and all 16 members complete at
those budgets. A prior run sharing 256 leaves across all members reached only
54.93%, so member-wise allocation resolves that resource artifact. The final run
uses 60.8 CPU-minutes and 30.5 wall-minutes on two workers. All successful leaves
come from local branch enumeration; the affine layer closes none.

The report consists of 16 immutable shards and passes independent config,
selection, source-manifest, member-row, hash, and exact area-accounting checks.
It fixes float32 state arithmetic and zero output delay. Because the input was
selected during label-free development, the result demonstrates sound family
breadth and resource scaling, not population coverage.

Authoritative artifacts:

- `cartesian_member_scaling_sound_v5_summary.json`
- `cartesian_member_scaling_sound_v5_rows.csv`
- `../../artifacts/shd_v82_cartesian_member_scaling_concurrent_v1/cartesian_member_scaling.json`
- `../../artifacts/shd_v82_cartesian_member_scaling_concurrent_v1/artifact_verification.json`
- `../../paper/figures/shd_cartesian_member_scaling_sound_v5.pdf`

```powershell
python experiments/select_shd_cartesian_development.py --output-root artifacts/shd_v71_cartesian_development_selection_canonical_v1
python experiments/run_shd_hybrid_cartesian_member_scaling.py --config configs/experiments/shd_hybrid_cartesian_member_scaling_v2.json --output-root artifacts/shd_v82_cartesian_member_scaling_concurrent_v1
python experiments/aggregate_shd_hybrid_cartesian_member_scaling.py --report artifacts/shd_v82_cartesian_member_scaling_concurrent_v1/cartesian_member_scaling.json --output results/shd_v1/cartesian_member_scaling_sound_v5_summary.json --rows results/shd_v1/cartesian_member_scaling_sound_v5_rows.csv --figure paper/figures/shd_cartesian_member_scaling_sound_v5.pdf
python experiments/verify_shd_cartesian_member_scaling_artifact.py --report artifacts/shd_v82_cartesian_member_scaling_concurrent_v1/cartesian_member_scaling.json --config configs/experiments/shd_hybrid_cartesian_member_scaling_v2.json --output artifacts/shd_v82_cartesian_member_scaling_concurrent_v1/artifact_verification.json
```

### Post-repair bounded-family headroom

`shd_repair_family_grid_full_audit_reset_clustered_v3_summary.json` compares the cast-faithful
reset-target repaired models against the original source/reference predictions
over a 9-by-9 local family using all 861 audit inputs per seed.
Certificate-directed repair raises sampled family identity from 63.88% to 76.14%
and improves every seed, while logit-only reaches 76.33%, global threshold
scaling 70.41%, and the labeled target fine-tune 75.31%. It supersedes the
full-float32 `repair_family_grid_summary.json` and 128-input diagnostics.
Its proposed-minus-logit mean is -0.19 points, with a 95% input-cluster
bootstrap interval of [-1.51, 1.14] points and a five-seed interval of
[-3.05, 2.68].

`shd_repair_family_grid_full_audit_floor_clustered_v3_summary.json` applies the same
cast-faithful diagnostic to the floor-rounded fixed-point condition. Sampled
family identity rises from 18.58% without repair to 75.42% after
certificate-directed repair, versus 76.52% for logit-only, 21.07% for global
threshold scaling, and 79.42% for the labeled target fine-tune. The proposed
method improves every seed but the result is not a sound continuous certificate:
fixed-point local branching remains unimplemented.
The proposed-minus-logit mean is -1.09 points; its input-cluster interval is
[-2.42, 0.23] and its seed-level interval is [-3.33, 1.14].

`repaired_branch_set_cells_summary.json` asks whether the sampled improvement
makes sound explicit branch analysis easier. It does not: no representative
cell completes, and at 128 partitions per axis the mean cell-median abort step
falls from 47.0 without repair to 42.83 for certificate-directed repair and
40.33 for logit-only. The present methods are transport repairs, not
certificate-restoring repairs.

```powershell
python experiments/run_shd_repair_family_grid.py --seed 1701
python experiments/aggregate_shd_repair_family_grid.py
python experiments/run_shd_repaired_branch_set_cells.py --seed 1701
python experiments/aggregate_shd_repaired_branch_set_cells.py
```

### Pointwise guard-margin repair kill test

`guard_margin_development_summary.json` records a calibration-only, single-seed
development decision. The selected 40-epoch run recovers 79.6% of reset loss
and reaches 73.44% identity over the sampled 9-by-9 family. Its finest sound
branch-set diagnostic reaches mean abort timestep 44.42, above 42.83 for the
earlier certificate-directed repair but below 47.0 without repair, and no cell
completes. We therefore do not expand the pointwise penalty to five seeds.

The immutable summary is regenerated from ignored development artifacts with:

```powershell
python experiments/aggregate_shd_guard_margin_development.py
```

### Sampled-family repair kill test

`sampled_family_repair_development_summary.json` compares worst-point training
on four corners and on a 3-by-3 timestep/threshold grid. The best single-seed
checkpoint uses the 3-by-3 grid for 10 epochs and reaches 75.00% identity on the
disjoint 9-by-9 audit grid. This is one of 128 inputs above certificate-directed
repair (74.22%) but below logit-only (77.34%). At 40 epochs, calibration family
disagreements improve from 131 to 106 while audit-grid identity falls to 71.88%.
The method therefore fails the gate for sound proof analysis and five seeds.

Regenerate the immutable development summary from ignored artifacts with:

```powershell
python experiments/aggregate_shd_sampled_family_repair.py
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
