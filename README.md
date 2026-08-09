# PINES

**Prediction Invariance across Neuromorphic Execution Semantics for Spiking Neural Networks**

PINES is a clean-room research implementation for prospective, label-free
certification and repair of finite-horizon digital SNN deployments. It defines
execution semantics as executable data, compares a readable scalar interpreter
against a vectorized emulator, produces immutable prediction-preservation
certificate artifacts, and keeps emulator and physical-hardware uncertainty
separate.

The first research release implements:

- versioned execution semantics for integration, threshold/reset ordering,
  fixed-point arithmetic, delays, and randomness;
- scalar, vectorized NumPy, and vectorized PyTorch dense recurrent LIF execution;
- exact checking over finite semantics families and exhaustive small-state
  counterexample search;
- exact one-sided Clopper--Pearson disagreement bounds with simultaneous
  confidence correction and physical conformance composition;
- label-free, certificate-directed threshold/leak calibration with disjoint
  calibration and audit sets;
- immutable JSON reports, command-line workflows, optional NIR/SpiNNaker2
  adapters, and a fresh parameterized RTL neuron;
- a frozen evaluation protocol and reproducible result artifacts.

The code does **not** claim that arbitrary update orderings are equivalent.
Equivalence is reported only when a mapping is proved or a declared finite
state/input domain is exhausted without a counterexample.

## Quick start

```powershell
python -m pip install -e .
python -m pytest
python experiments/run_toy_study.py --output artifacts/toy-study
pines --help
```

Physical certificates require an unlabeled canary run and a complete hardware
manifest. Without one, reports are explicitly conditional on the emulator.

## Scope and status

This repository covers finite-horizon digital classification SNNs. Analog
mismatch, continual learning, control, and energy claims are out of scope.
The current family engine is exact for an explicitly enumerated bounded family;
its relational affine/polygon/branch engine also proves targeted continuous
timestep--threshold families for recurrent float-state models. A frozen
five-seed confirmation audit is being recomputed after proof-integrity hardening.
The first completed seed certifies 347/861 SHD inputs (40.30%); the other four
seeds and canonical grid/Sobol falsification remain in progress. The previous
1,398/4,145 pooled headline and its binomial interval are superseded.
Separately, a source-hash-verified member-wise analysis certifies all 16
integration/timing/reset/synaptic-delay combinations over a joint plus/minus 1%
timestep--threshold box for one label-free selected SHD input at 256 polygon
leaves per member. This establishes full-family feasibility and identifies
proof-budget bottlenecks, but it is selected-input development evidence rather
than population coverage.
Broader numeric-axis coverage, DVS full-model certification, and the two
physical campaigns remain gated research milestones rather than represented as
completed results.
