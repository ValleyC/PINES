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
- label-free transport calibration and mandatory recertification with disjoint
  calibration and audit sets;
- immutable JSON reports, command-line workflows, optional NIR/SpiNNaker2
  adapters, and a fresh parameterized RTL neuron;
- a frozen evaluation protocol and reproducible result artifacts.

The code does **not** claim that arbitrary update orderings are equivalent.
Equivalence is reported only when a mapping is proved or a declared finite
state/input domain is exhausted without a counterexample.

## Quick start

```powershell
python -m pip install -e ".[dev,benchmarks,torch]"
python -m pytest
python experiments/run_toy_study.py --output artifacts/toy-study
pines --help
```

Install `.[datasets]` only when rebuilding the raw event-dataset caches. Tonic
1.6 requires NumPy below 2, so the resolver will select a compatible NumPy
version for that preprocessing environment. The recorded manuscript runs use
the hash-verified processed arrays and their separately recorded environment.

Physical certificates require an unlabeled canary run and a complete hardware
manifest. Without one, reports are explicitly conditional on the emulator.

## Scope and status

This repository covers finite-horizon digital classification SNNs. Analog
mismatch, continual learning, control, and energy claims are out of scope.
The current family engine is exact for an explicitly enumerated bounded family.
Its relational affine, polygon, and local-branch engine also proves targeted
continuous timestep and threshold families for recurrent float-state models.
The soundness-corrected five-seed SHD reset-family audit is complete over 4,305
model-input pairs. It certifies 33.89%, finds explicit grid or Sobol
counterexamples for 36.84%, and leaves 29.27% unresolved. The simultaneous
software bound is 70.56 points on average, so every seed rejects the one, two,
and five-point budgets. Independent 9 by 9 grid and 1,024-point Sobol searches
find no contradiction to any certified input. Median analysis time is 51.46
seconds per input.

A matched five-seed width study fixes 80 model-input pairs and all proof
budgets while jointly varying timestep and threshold uncertainty. Certified
fractions are 73.75%, 68.75%, 37.50%, and 0% at plus or minus 0.25%, 0.5%, 1%,
and 2%. Independent 9 by 9 grids find concrete changes for 26.25%, 28.75%,
32.50%, and 42.50%. The remaining stable but unresolved fractions are 0%,
2.50%, 30.00%, and 57.50%, which separates observed semantic instability from
proof or resource conservatism.

Exact finite-family enumeration certifies 38.72% of SHD inputs, 57.69% of DVS
Gesture inputs, and 97.67% of N-MNIST inputs. The corresponding simultaneous
mean bounds are 65.90, 56.19, and 2.93 points. Only the feedforward N-MNIST
control accepts the five-point budget for every seed. A separate
source-hash-verified analysis certifies all 16 discrete members over a joint
plus or minus 1% timestep and threshold box for one label-free selected SHD
input at 256 polygon leaves per member. This is selected-input development
evidence rather than population coverage.

Matched repair experiments recover more than 70% of the lost accuracy on
average in the three reported conditions, but the certificate-directed
objective does not consistently beat logit-only distillation and no repaired
model obtains a five-point certificate. DVS Gesture remains development
evidence because official test accuracy influenced reference-pipeline
selection. The two physical campaigns remain incomplete.
