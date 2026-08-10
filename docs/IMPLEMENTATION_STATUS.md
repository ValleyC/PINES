# Implementation status — 2026-08-09

## Evidence update, 2026-08-09

All planned non-hardware result cells for the six-page manuscript are now
filled. The soundness-corrected SHD continuous reset-family audit covers 4,305
model-input pairs across five seeds. It certifies 1,459 pairs, or 33.89%. The
union of an independent 9 by 9 grid and a 1,024-point Sobol design finds
counterexamples for 1,586 noncertified pairs, or 36.84%, and leaves 1,260 pairs,
or 29.27%, unresolved. Neither search contradicts a certified input. Exact
fraction and leaf accounting passes an independent verifier over all 500 shard
artifacts. Mean simultaneous semantic risk is 70.56 points and median analysis
time is 51.46 seconds per input.

A reviewer-driven continuous-width study holds 80 SHD model-input pairs, five
seeds, and every proof budget fixed while widening joint timestep and threshold
uncertainty. At plus or minus 0.25%, 0.5%, 1%, and 2%, the certified,
grid-falsified, and stable-but-unresolved fractions are 73.75/26.25/0,
68.75/28.75/2.50, 37.50/32.50/30.00, and 0/42.50/57.50 percent. Median runtime
is 3.72, 16.97, 48.42, and 24.63 seconds per input. The lower final runtime
reflects early budget exhaustion rather than improved tractability. This study
quantifies both genuine semantic changes and the analyzer's practical
tightness limit as the declared contract widens.

Exact finite-family results are complete on the full frozen audits. Mean
certified fractions are 38.72% for SHD, 57.69% for DVS Gesture, and 97.67% for
N-MNIST. Mean simultaneous bounds are 65.90, 56.19, and 2.93 points. All five
N-MNIST seeds accept the five-point budget. The recurrent tasks reject all
three budgets because the declared family contains many genuine prediction
changes.

The matched repair matrix is also complete. Certificate-directed repair
recovers 72.1% and 89.7% on the two SHD targets and 91.1% on DVS Gesture. It
does not consistently outperform logit-only distillation, and no method
restores a five-point certificate. The repair result therefore supports a
post-rejection opportunity and mandatory recertification, not superiority of
the current repair objective.

The software evidence supports the semantics-contract and sound-certificate
formulation. It does not pass the full submission contract. Physical
conformance is absent, recurrent fixed-target slack exceeds the frozen
tightness gates, DVS Gesture remains development evidence, and a clean external
reproduction package is not yet complete.

## Implemented and verified

- Versioned operational semantics and JSON schema.
- Scalar, vectorized NumPy, and vectorized PyTorch dense recurrent LIF execution.
- Deterministic fixed-point quantization, rounding, saturation/wrap, and delays.
- Exact finite-family certification and sound interval semantics boxes with reset
  branch splitting and merging.
- Structural identity and exhaustive finite-input trace checking for small
  networks, including an update-order counterexample.
- Exact one-sided Clopper--Pearson limits, simultaneous family correction,
  zero-error sample planning, and emulator/hardware composition.
- Differentiable certificate-mass repair plus hard per-neuron threshold, tau,
  bias, and incoming-scale refinement with strict calibration/audit separation.
- Immutable report contracts, command-line workflows, NIR and SpiNNaker2
  conversion adapters, hardware manifest validation, fresh Virtex-7 RTL, and 122
  passing software tests.
- Matched repair baselines with explicit label, sample, optimizer-step,
  trainable-parameter, runtime, selection, split, and artifact provenance.

## Completed software evidence

- Five-seed SHD, N-MNIST, and DVS Gesture development matrices under ten frozen
  semantic targets. The DVS result still requires a fresh external sequestered
  replication because its reference pipeline was selected using official test
  accuracy before semantic targets were run.
- Five-seed SHD matched repair study on reset-to-value and floor-rounded
  fixed-point execution under the cast-faithful executor. Certificate-directed
  repair recovers 72.1% and 89.7% of lost accuracy, respectively; nine of its
  ten cells clear 70%, with reset seed 8119 recovering only 23.1%. It beats no
  repair and global scaling in every cell, but logit-only has higher mean
  recovery in both conditions and labeled QAT is strongest on the genuine
  fixed-point comparison. No repaired method/seed/condition cell accepts a
  five-point budget.
- Five-seed DVS Gesture floor-rounded repair development study. Restricted
  certificate-directed calibration recovers 91.1% of lost accuracy with every
  seed above 79.5%, versus 89.3% for logit-only, 74.8% for global scaling, and
  99.6% for 97-label QAT. It beats logit-only in only three paired seeds and its
  21.5-point mean bound is substantially looser than logit-only's 15.1 points.
- Full-audit SHD finite-family analysis: 38.72% mean member-wise coverage over
  the 16-member family, exactly matching enumerated agreement and passing the
  declared 20% gate. The preserved 18.79% predecessor merged logits across
  mutually exclusive semantics and is superseded for the finite-family claim.
- N-MNIST static diagnostic on deterministic 128-input subsets: 98.59% mean
  member-wise coverage, equal to exact finite-family agreement.
- Within-SHD recurrence intervention: zeroing only recurrent feedback raises
  both static coverage and exact agreement from 38.72% to 89.31%. This is a
  mechanism diagnostic, not an accuracy-matched comparison.
- Five-seed horizon diagnostic: trained recurrent finite-family coverage rises
  from 30.94% at 10 bins to 36.17% at 50 and tracks exact agreement. Semantic
  instability does not monotonically worsen with unroll length.
- Full-audit continuous-family analysis: joint timestep and threshold uncertainty
  of only plus/minus 1% yields zero soundly certified inputs for every SHD seed.
  Endpoint-grid agreement remains 36.9%, demonstrating severe abstraction slack.
- Five-seed reference-member radius sweep: joint coverage falls from 89.64% at
  relative radius 1e-6 to 50.57% at 1e-5 and zero at 1e-4. An 8-by-8 uniform
  partition at plus/minus 1% still covers zero of 128 inputs; a 16-by-16 grid
  covers zero of 64 after 89 seconds. Useful uniform cells would imply about one
  million continuous boxes before the 16 discrete semantics are included.
- Five-seed fixed-branch endpoint diagnostic: joint center-trace identity falls
  from 99.28% at radius 1e-6 to 54.12% at 1e-4, 0.74% at 1e-3, and zero at 1e-2,
  while one-percent corner prediction identity remains 89.36%. A fixed-center-
  branch affine analyzer may help at tiny radii but has no coverage headroom for
  the intended plus/minus 1% box.
- Five-seed branch-complexity diagnostic: at a 17-by-17 reference-member grid,
  inputs exhibit 81.67 distinct spike traces on average and none retains a
  single trace, yet 80.94% retain one prediction and only 4.35% of grid-center
  pairs disagree. Internal branch proliferation is much stronger than decision
  instability.
- Five-seed full-family falsification grid: a 9-by-9 continuous grid times all
  16 discrete members retains 31.09% prediction identity on deterministic
  128-input subsets, with every seed above the 20% target. A better sound
  abstraction therefore has only about eleven points of sampled mean headroom,
  but is not ruled out by true instability.
- Sound explicit branch-set kill test: separating every possible spike vector
  fails to complete representative 50-step cells even with a 65,536-state cap
  and 128-by-128 parameter partitioning. Merging equivalent spike/logit states
  and retaining all states fail at essentially the same timestep.
- Sound decision-margin domain kill test: directly propagating pairwise output
  margins is strictly tighter on a constructed shared-spike witness, but matches
  ordinary interval coverage at zero on staged SHD reset-family audits through
  32-by-32 partitioning. Hidden recurrent state/guard correlation, rather than
  independent output bounds, is the dominant unresolved loss.
- Sound adaptive-cover kill test: 80 of 128 frozen audit-pool inputs are stable
  on a finite 9-by-9 reset-family grid, but adaptive margin branch-and-bound
  retires no sub-box for the first two through 256 leaves and still covers 0% of
  the first input after 8,191 analyses and 4,096 leaves. Adaptive scheduling
  cannot rescue the non-relational recurrent-state domain.
- Sound relational affine-guard advance: shared timestep/threshold generators
  through recurrent hidden states yield the first nonzero SHD local proof
  coverage—0.15% at 4-by-4, 3.74% at 8-by-8, and 18.46% at 16-by-16, versus zero
  for decision intervals. Adaptive affine coverage on a grid-stable input rises
  to 95.61% at 16,384 leaves, but unresolved guard-surface cells prevent a full
  certificate. This method advances to symbolic guard cuts, not five seeds yet.
- Matched guard-guided affine split test: uncertain-guard sensitivity raises
  certified volume from 61.47% to 62.84% at 1,024 leaves and from 85.72% to
  86.35% at 4,096, but is slightly worse at 256. The small, non-monotone gain
  rejects further axis-policy tuning and supports non-axis-aligned constraints.
- Sound polygonal recurrent-guard advance with explicit float32 roundoff:
  current, state, output, and logit conversions receive conservative rounding
  envelopes while direct margins retain shared-spike cancellation. Staged guard
  cuts certify 82.26% at 1,024 leaves, 99.7557% at 4,096, and 99.8331% at 16,384
  for one finite-grid-stable SHD input. The largest run leaves 0.1669% area in
  13,158 polygons. Four times the final leaf budget improves coverage by only
  0.077 points, so no certificate is issued and leaf escalation stops. The
  earlier 99.9477% arithmetic-idealized summary is superseded.
- Exact threshold-boundary slice oracle: for reset-to-value at fixed timestep,
  it exhausts every representable binary64 threshold scale using contiguous
  spike-trace cells and the emulator's actual comparisons. All 257 timestep
  slices for the same SHD development input preserve the reference class; 2,716
  cells are sufficient in 12.1 seconds, with at most 17 in one slice. This is
  not a proof between timestep slices, so the joint certificate remains open.
- Hashed residual-geometry export: the corrected 4,096-leaf cover retains 3,365
  unresolved convex polygons in a compact archive. Their median actual
  timestep-factor width is 4.19e-5, below the 7.8125e-5 exact-slice spacing, and
  1,271 polygons intersect no exact slice. Their projections span both full
  axes, rejecting more uniform slice densification and requiring a solver that
  consumes the polygon constraints directly.
- Residual-targeted counterexample search: a validated GPU parameter-batch
  executor evaluates every polygon vertex, edge midpoint, and centroid plus 32
  deterministic random convex points per polygon. All 128,460 unique executions
  retain class 2 and the minimum reference-class margin is 1.033. This supports
  abstraction slack but is explicitly not used as a certificate; further random
  search is stopped in favor of the sound joint oracle.
- Sound local polygon branch closure: conditional affine propagation enumerates
  both outcomes of every guard that remains uncertain, retaining even infeasible
  assignments. All 3,365 residual polygons complete and predict only class 2;
  median peak branch count is 2, the 99th percentile is 3, and the maximum is
  14. Together with the affine cover, total parameter area is one within 6.8e-12,
  yielding the first complete per-input joint continuous-family certificate.
- Frozen five-seed hybrid screen: with 64 leaves, 64 local branches, and eight
  guard cuts fixed before opening the screen, 58/160 untouched SHD audit inputs
  are fully certified (36.25%), and every seed is nonzero. Because the same 32
  input positions recur across seeds, no pooled binomial interval is reported.
  The historical 9-by-9 grid contradicts none of the 58 certificates,
  finds concrete flips for 48 inputs, and is stable but inconclusive for 54. The
  declared 20% advancement gate passes and triggers the all-input audit; both
  screen and grid are retained only as development provenance.
- Soundness-corrected five-seed audit: all 4,305 frozen model--input evaluations
  are being recomputed with input-drive reduction envelopes, outward branch
  logits, shared polygon intersections, no positive-area sliver deletion, and
  outward partial-area accounting. Seed 1701 is complete at 347/861 certificates
  (40.30%) and 61.33% mean proved volume. Relative to the superseded analyzer,
  345 certificates remain, three are withdrawn, and two are gained. The remaining
  seeds, independent artifact verification, and canonical falsification are live.
- The prior 1,398/4,145 coverage headline, pooled binomial interval, and
  grid/Sobol counts are superseded. Final uncertainty will separately report an
  input-cluster bootstrap and a five-training-seed interval.
- Broader family engine: the local oracle now carries explicit synaptic/output
  queues, encloses exponential Euler with a second-order remainder, and conjoins
  mutually exclusive discrete members. Randomized recurrent differential tests
  cover these additions; fixed-point local branching remains open.
- Selected-input 16-member continuous certificate: a canonical, label-free
  $5\times5$ development screen finds 236/800 stable calibration inputs and
  selects dataset index 862 by its minimum reference-class margin. Member-wise
  sound analysis covers all combinations of forward/exponential Euler,
  pre/post-integration threshold timing, subtractive/reset-to-value reset, and
  zero/one-step synaptic delay over the full plus/minus 1% timestep--threshold
  box. Mean proved area rises from 15.47% at 16 leaves per member to 70.97% at
  64 and 100% at 256; all 16 members complete at the final budget. The run uses
  60.8 CPU-minutes (30.5 wall-minutes on two workers), while the slowest member
  takes 305.6 seconds. The affine layer closes no leaf; every success requires
  sound local branch enumeration. All 16 immutable shards pass independent
  config, source-hash, row, and exact area-accounting verification. This closes
  semantic breadth for one selected float32 input with zero output delay, not
  population coverage or the numeric-format axis.
- Five-seed bounded-subfamily decomposition: after adding the 9-by-9 continuous
  grid, reset retains 60.31% sampled identity, integration 68.91%, timing and
  delay each 69.06%, reset plus delay 46.72%, integration plus timing 55.47%,
  and the full family 31.09%. Every primary-axis seed remains above 56.25%.
- Five-seed post-repair reset-family diagnostic: certificate-directed repair
  raises sampled family identity from 63.88% to 76.14% on all 861 frozen audit
  inputs per seed and improves every seed, but logit-only reaches 76.33%. These
  values use the cast-faithful executor and repaired models and supersede the
  earlier full-float32 and 128-input diagnostics. The proposed-minus-logit
  paired input-cluster interval is [-1.51, 1.14] points and the seed-level
  interval is [-3.05, 2.68], so neither objective is superior. A
  single-seed sound branch-set test on the earlier models completes
  no representative repaired-model cell; at the finest partition, repair
  reduces rather than increases the reached proof depth.
- Five-seed floor-fixed-point family diagnostic: sampled 9-by-9 identity rises
  from 18.58% without repair to 75.42% after certificate-directed repair on all
  861 audit inputs per seed, a 56.84-point paired gain that is positive in every
  seed. Logit-only reaches 76.52% and the labeled target fine-tune 79.42%. This
  condition's proposed-minus-logit intervals are [-2.42, 0.23] points over input
  clusters and [-3.33, 1.14] over seeds. This is not yet a sound
  continuous certificate because fixed-point local branching is unimplemented.
- Single-seed proof-aware repair kill test: a selected pointwise guard-margin
  penalty reaches timestep 44.42 at 128-by-128 partitioning, partially above
  certificate-directed repair's 42.83 but below 47.0 without repair, and
  completes no cell. Stronger weights reduce transport recovery without
  improving proof depth, so this route is not expanded to five seeds.
- Single-seed sampled-family repair kill test: worst-point training on four
  corners or a 3-by-3 grid restores accuracy but does not reliably improve a
  disjoint 9-by-9 family. The best checkpoint reaches 75.00% identity, one of
  128 inputs above certificate-directed repair but below logit-only's 77.34%.
  Longer training improves calibration disagreements while audit identity drops
  to 71.88%, so sound-analysis and five-seed expansion are skipped.
- Sample-complexity analysis: under ten-way simultaneous 95% confidence, zero
  disagreement requires 528, 263, and 104 pairs for one-, two-, and five-point
  budgets. SHD and N-MNIST exceed all three thresholds; DVS does not exceed the
  first two.
- Disagreement sharpness audit: adversarial label constructions attain both
  signs of the empirical disagreement bound for all 150 stored benchmark,
  seed, and target prediction pairs. No uniformly tighter distribution-free
  accuracy-change bound is possible from paired predictions alone.

## Current scientific verdict

The evidence supports execution-semantic transport as a real recurrent-SNN
problem, disagreement as a strong label-free ranking signal, label-free repair
as an accuracy-recovery mechanism, and sound prospective certification of both
finite enumerated families and a targeted joint continuous family. The local
branch result removes the previous central scientific blocker: a complete
continuous certificate now exists. A separate selected-input result also closes
all 16 declared integration/timing/reset/synaptic-delay members over the same
continuous box. The soundness-corrected five-seed targeted-family audit is in
progress; its completed first seed retains 40.30% full-box coverage after paired
withdrawals and gains. A final population estimate and canonical falsification
claim await all five seeds, and population coverage of the complete Cartesian
family has not been measured.

This advance is narrower than the complete manuscript contract. The population
audit covers reset-to-value with joint plus/minus 1% timestep and threshold
uncertainty, forward Euler, float32 state arithmetic, and zero delay. The
selected-input Cartesian certificate adds exponential integration, threshold
timing, subtractive reset, and synaptic delay, but has not passed a frozen
population audit. Fixed-point local branching, DVS family certification, and
both physical conformance terms remain open.

Targeted bounded-axis contracts remain scientifically viable: their sampled
ceilings are 60--69%, and repair can add roughly ten points of reset-family
headroom. However, the current method restores predictions rather than static
certifiability. It must be described as transport repair, not certificate
restoration, until a sound post-repair family bound succeeds.

The distribution certificate is mathematically valid but cannot be made
uniformly tighter without adding information or assumptions. A top-venue claim
must therefore specify and validate such an assumption rather than presenting
raw label-free disagreement as both universal and operationally tight.

The manuscript framing therefore advances from ``missing certificate method'' to
``incomplete breadth and physical validation.'' It remains conditional on at
least one sound bounded family on DVS Gesture, two
physical backends, emulator--hardware conformance, and certificate-directed
repair that improves post-repair certified risk rather than only accuracy. The
distribution certificate also remains mathematically sharp but operationally
loose without additional assumptions.

## Open experimental milestones

- Complete, provenance-check, aggregate, and canonically falsify the
  soundness-corrected five-seed targeted-family SHD audit.
- Improve or preregister the resource allocation needed for a frozen
  population audit of the 16-member continuous Cartesian family. The selected
  development input is fully certified, but its 60.8 CPU-minute cost cannot be
  extrapolated into a tractability claim for all audit inputs.
- Extend the sound hybrid domain to the DVS convolutional/recurrent model or
  define and preregister an equivalently faithful reduced abstract interface.
- Multi-step family-verified repair margins. A pointwise center-execution guard
  penalty and worst-loss objectives over sparse execution grids are now
  empirically ruled out as certificate-restoring objectives.
- Full-training-set supervised retraining as a separately budgeted oracle; the
  completed matched baseline intentionally uses only 800 labels and 280 updates.
- Fresh sequestered DVS Gesture replication.
- Cocotb/formal execution in an installed HDL toolchain and board synthesis.
- Actual SpiNNaker2 and Virtex-7 canary captures and repeated-run analysis.
- Certificate-directed repair across both physical backends.
- External clean-machine reproduction of every software figure and table.

No physical evidence is currently available on this machine: neither target
board is attached, and the SpiNNaker2 and HDL execution dependencies are absent.
