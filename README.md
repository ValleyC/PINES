# Transport Certificates for SNNs

This is a clean-room research implementation for prospective, label-free
certification and repair of finite-horizon digital SNN deployments. It defines
execution semantics as executable data, compares a readable scalar interpreter
against a vectorized emulator, produces immutable certificate artifacts, and
keeps emulator and physical-hardware uncertainty separate.

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
- a preregistered evaluation protocol and anonymous manuscript scaffold.

The code does **not** claim that arbitrary update orderings are equivalent.
Equivalence is reported only when a mapping is proved or a declared finite
state/input domain is exhausted without a counterexample.

## Quick start

```powershell
python -m pip install -e .
python -m pytest
python experiments/run_toy_study.py --output artifacts/toy-study
transportcert --help
```

Physical certificates require an unlabeled canary run and a complete hardware
manifest. Without one, reports are explicitly conditional on the emulator.

## Scope and status

This repository covers finite-horizon digital classification SNNs. Analog
mismatch, continual learning, control, and energy claims are out of scope.
The current family engine is exact for an explicitly enumerated bounded family;
full-model zonotope propagation and the two physical campaigns are tracked as
gated research milestones rather than represented as completed results.
