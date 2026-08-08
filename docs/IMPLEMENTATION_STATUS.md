# Implementation status — 2026-08-08

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
  conversion adapters, hardware manifest validation, fresh Virtex-7 RTL, and 70
  passing software tests.
- Matched repair baselines with explicit label, sample, optimizer-step,
  trainable-parameter, runtime, selection, split, and artifact provenance.

## Completed software evidence

- Five-seed SHD, N-MNIST, and DVS Gesture development matrices under ten frozen
  semantic targets. The DVS result still requires a fresh external sequestered
  replication because its reference pipeline was selected using official test
  accuracy before semantic targets were run.
- Five-seed SHD matched repair study on reset-to-value and floor-rounded
  fixed-point execution. Certificate-directed repair recovers 77.2% and 87.8%
  of lost accuracy, respectively, and every one of its ten cells clears 70%.
  It beats no repair and global scaling in every cell, but loses to logit-only
  on floor rounding and to labeled QAT on the genuine fixed-point comparison.
  No repaired method/seed/condition cell accepts a five-point budget.
- Five-seed DVS Gesture floor-rounded repair development study. Restricted
  certificate-directed calibration recovers 91.1% of lost accuracy with every
  seed above 79.5%, versus 89.3% for logit-only, 74.8% for global scaling, and
  98.7% for 97-label QAT. It beats logit-only in only three paired seeds and its
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
as an accuracy-recovery mechanism, and prospective certification of a finite
enumerated execution family. The proposed repair objective is not uniformly
better than logit-only imitation, and repair does not make the distribution
certificate operationally tight. The evidence also does not support the intended
continuous bounded-family claim with the current interval abstraction.
Recurrence causes much of the true finite-family instability, while continuous
interval relaxation and decision-identity loss remain substantial.

The distribution certificate is mathematically valid but cannot be made
uniformly tighter without adding information or assumptions. A top-venue claim
must therefore specify and validate such an assumption rather than presenting
raw label-free disagreement as both universal and operationally tight.

The NeurIPS framing remains conditional on a materially tighter recurrence-aware
sound abstraction and prospective physical conformance. If that does not reverse
the continuous-family gate, the project follows the declared pivot to a narrower
finite-family or systems/risk-ranking contribution.

## Open experimental milestones

- Tighter recurrence-aware analysis, such as branch budgeting, zonotopes, or a
  hybrid exact/abstract domain, evaluated against the existing exact-family gap.
- Full-training-set supervised retraining as a separately budgeted oracle; the
  completed matched baseline intentionally uses only 800 labels and 280 updates.
- Fresh sequestered DVS Gesture replication.
- Cocotb/formal execution in an installed HDL toolchain and board synthesis.
- Actual SpiNNaker2 and Virtex-7 canary captures and repeated-run analysis.
- Certificate-directed repair across both physical backends.
- External clean-machine reproduction of every software figure and table.

No physical evidence is currently available on this machine: neither target
board is attached, and the SpiNNaker2 and HDL execution dependencies are absent.
