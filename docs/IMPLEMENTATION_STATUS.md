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
  conversion adapters, hardware manifest validation, fresh Virtex-7 RTL, and 64
  passing software tests.

## Completed software evidence

- Five-seed SHD, N-MNIST, and DVS Gesture development matrices under ten frozen
  semantic targets. The DVS result still requires a fresh external sequestered
  replication because its reference pipeline was selected using official test
  accuracy before semantic targets were run.
- Five-seed SHD reset-to-value repair study. Certificate-directed repair recovers
  77.2% of lost accuracy, but its mean post-repair bound is 15.1 points and does
  not accept any seed at a five-point budget.
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
- Sample-complexity analysis: under ten-way simultaneous 95% confidence, zero
  disagreement requires 528, 263, and 104 pairs for one-, two-, and five-point
  budgets. SHD and N-MNIST exceed all three thresholds; DVS does not exceed the
  first two.

## Current scientific verdict

The evidence supports execution-semantic transport as a real recurrent-SNN
problem, disagreement as a strong label-free ranking signal, label-free repair
as an accuracy-recovery mechanism, and prospective certification of a finite
enumerated execution family. It does not support the intended continuous bounded-
family claim with the current interval abstraction. Recurrence causes much of
the true finite-family instability, while continuous interval relaxation and
decision-identity loss remain substantial.

The NeurIPS framing remains conditional on a materially tighter recurrence-aware
sound abstraction and prospective physical conformance. If that does not reverse
the continuous-family gate, the project follows the declared pivot to a narrower
finite-family or systems/risk-ranking contribution.

## Open experimental milestones

- Tighter recurrence-aware analysis, such as branch budgeting, zonotopes, or a
  hybrid exact/abstract domain, evaluated against the existing exact-family gap.
- Per-platform QAT and fully supervised target-retraining baselines at matched
  sample, optimization, and compute budgets.
- Fresh sequestered DVS Gesture replication.
- Cocotb/formal execution in an installed HDL toolchain and board synthesis.
- Actual SpiNNaker2 and Virtex-7 canary captures and repeated-run analysis.
- Certificate-directed repair across DVS Gesture and both physical backends.
- External clean-machine reproduction of every software figure and table.

No physical evidence is currently available on this machine: neither target
board is attached, and the SpiNNaker2 and HDL execution dependencies are absent.
