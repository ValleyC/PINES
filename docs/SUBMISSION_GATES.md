# Submission gates

The submission proceeds only if every item below is true at the six-week freeze:

- SpiNNaker2 and Virtex-7 results exist for SHD and DVS Gesture.
- At least four preregistered target conditions lose more than five accuracy
  points before repair.
- The frozen primary matrix has no violation of the simultaneous 95% total
  bound.
- Median total-bound slack is at most three points and the 90th percentile at
  most eight points.
- Hardware conformance does not make the total certificate vacuous.
- Label-free repair recovers at least 70% of loss across both tasks and both
  backends, beating global threshold scaling and PTQ at matched cost.
- Every exact-equivalence claim survives symbolic derivation, exhaustive
  small-state checking, and adversarial trace search.
- A clean machine reproduces every software figure/table; physical artifacts
  include raw traces and firmware/bitstream identities.

If static family bounds are vacuous on more than 80% of inputs or physical
conformance dominates, the project pivots to a systems venue. The guarantee is
not weakened to preserve a submission narrative.

## Current gate audit, 2026-08-09

- Physical backend gate: not evaluated.
- Large-loss gate: satisfied in the software matrix.
- Software bound-validity gate: satisfied with zero observed violations.
- Slack gate: failed on SHD and DVS Gesture, passed on N-MNIST.
- Hardware conformance gate: not evaluated.
- Repair gate: mean recovery exceeds 70% in the three reported conditions, but
  cross-backend evidence is unavailable and the current objective is not
  consistently stronger than logit-only distillation.
- Exact-equivalence gate: structural identity and finite-state counterexample
  checks pass. No nontrivial closed-form equivalence is claimed.
- Clean-machine artifact gate: not yet satisfied.

The manuscript is therefore not ready for submission. The core formulation
remains supported, while repair superiority, tight recurrent acceptance, and
physical guarantees must not be claimed from the current evidence.
