# Implementation status — 2026-08-07

## Complete in the foundation release

- Versioned operational semantics and JSON schema.
- Scalar, vectorized NumPy, and vectorized PyTorch dense recurrent LIF execution.
- Deterministic fixed-point quantization, rounding, saturation/wrap, and delays.
- Exact finite-family certification and sound interval semantics boxes with
  reset branch splitting and merging.
- Structural identity and exhaustive finite-input trace checking for small
  networks, including an update-order counterexample.
- Exact one-sided Clopper--Pearson limits, simultaneous family correction, and
  emulator/hardware composition.
- Differentiable certificate-mass repair plus hard per-neuron threshold, tau,
  bias, and incoming-scale refinement with
  strict calibration/audit ID separation and post-selection re-certification.
- Immutable report contracts, CLI, NIR and SpiNNaker2 conversion adapters,
  hardware manifest validation, fresh Virtex-7 RTL, and automated tests.

## Open experimental milestones

- DVS Gesture and N-MNIST architecture/training pipelines. The SHD recurrent
  pipeline, canonical source verification, frozen split, and semantic matrix are
  implemented; its five-seed evidence matrix is in progress.
- Optional exact MILP checker and zonotope domain beyond the implemented interval
  domain; exhaustive checking is currently the exact small-network route.
- Per-platform QAT and supervised retraining baselines.
- Cocotb/formal execution in an installed HDL toolchain and board synthesis.
- Actual SpiNNaker2 and Virtex-7 canary captures.
- Five-seed primary matrix, figures, tables, and external clean-machine replay.

These items are intentionally not represented as completed evidence.
