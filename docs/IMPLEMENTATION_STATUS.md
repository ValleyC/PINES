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
  conversion adapters, hardware manifest validation, fresh Virtex-7 RTL, and 63
  passing software tests.

## Completed software evidence

- Five-seed SHD, N-MNIST, and DVS Gesture development matrices under ten frozen
  semantic targets. The DVS result still requires a fresh external sequestered
  replication because its reference pipeline was selected using official test
  accuracy before semantic targets were run.
- Five-seed SHD reset-to-value repair study. Certificate-directed repair recovers
  77.2% of lost accuracy, but its mean post-repair bound is 15.1 points and does
  not accept any seed at a five-point budget.
- Full-audit SHD static-family analysis: 18.79% mean coverage over the 16-member
  family and 81.21% vacuity, which fails the declared recurrent-task gate.
- N-MNIST static diagnostic on deterministic 128-input subsets: 96.72% mean
  coverage and 98.59% exact family agreement.
- Within-SHD recurrence intervention: zeroing only recurrent feedback raises
  static coverage from 18.79% to 67.20% and exact agreement from 38.72% to
  89.31%. This is a mechanism diagnostic, not an accuracy-matched comparison.
- Five-seed horizon diagnostic: trained recurrent coverage stays below 20% from
  10 through 50 bins and increases slightly as margins accumulate. Shortening
  the unroll therefore does not rescue the recurrent certificate.
- Sample-complexity analysis: under ten-way simultaneous 95% confidence, zero
  disagreement requires 528, 263, and 104 pairs for one-, two-, and five-point
  budgets. SHD and N-MNIST exceed all three thresholds; DVS does not exceed the
  first two.

## Current scientific verdict

The evidence supports execution-semantic transport as a real recurrent-SNN
problem, disagreement as a strong label-free ranking signal, and label-free
repair as an accuracy-recovery mechanism. It does not support the intended broad,
tight recurrent-family certificate with the current interval abstraction.
Recurrence causes much of the true family instability, while abstract relaxation
and decision-identity loss remain substantial after intervention and repair.

The NeurIPS framing remains conditional on a materially tighter recurrence-aware
sound abstraction and prospective physical conformance. If that does not reverse
the recurrent primary-task gate, the project follows the declared pivot to a
narrower per-axis or systems/risk-ranking contribution.

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
