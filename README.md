# PINES

**Prediction Invariance across Neuromorphic Execution Semantics for Spiking Neural Networks**

This repository is the anonymous review artifact for the PINES manuscript.
PINES analyzes whether a finite-horizon digital spiking neural network preserves
its predictions when execution semantics change across software and hardware
backends. The implementation includes executable semantics contracts, scalar
and vectorized emulators, finite and continuous-family certificate engines,
statistical disagreement bounds, label-free repair, and hardware conformance
interfaces.

Author names, affiliations, personal links, and citation metadata are omitted
during double-blind review.

## Repository map

- `src/pines/` contains the PINES implementation.
- `configs/` contains the frozen protocol and the configurations used by the
  retained manuscript experiments.
- `experiments/` contains the runners, aggregators, and evidence verifiers for
  the reported results.
- `results/` contains only the final software results used in the manuscript.
- `schemas/` defines the certificate, repair, semantics, and hardware
  manifest formats.
- `rtl/` contains the parameterized Virtex-7-oriented RTL and its verification
  harness.
- `hardware/` contains the physical-run manifest instructions.
- `hardware/bundles/` contains compact, attributed hardware handoff bundles
  for the frozen models and unlabeled canary inputs.
- `tests/` contains unit, differential, statistics, data-layout, and hardware
  arithmetic tests.

The manuscript-to-artifact map is in [`results/README.md`](results/README.md).

## Installation and checks

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[dev,torch]"
python -m pytest
python experiments/verify_manuscript_results.py
```

The lightweight end-to-end smoke study does not require benchmark data:

```powershell
pines reproduce --config configs/toy_study.json --output-dir artifacts/local-toy
```

The smoke study validates artifact creation and the command-line path. It is not
manuscript evidence.

## Manuscript evidence

The retained CSV and JSON files contain the final aggregate values reported in
Tables I through IV and Figure 2. Each summary records the source row file
and upstream reports. Large datasets, trained checkpoints, per-input prediction
arrays, and raw event caches are excluded from Git because of size and dataset
licensing. The frozen configurations identify those inputs.

Table V is reserved for physical SpiNNaker2 and Virtex-7 results. The software
repository includes backend adapters, RTL, schemas, and a hardware runbook, but
does not present uncollected hardware measurements as completed evidence.

The SHD FPGA handoff is documented in
[`hardware/bundles/shd_floor_q8q16_v1/README.md`](hardware/bundles/shd_floor_q8q16_v1/README.md).
It contains all five frozen source models, their final certificate-directed
repairs for the floor-rounded fixed-point target, 861 unlabeled audit inputs,
integer memory images, emulator predictions, cycle-level smoke traces, and
plain experiment manifests. Verify it with:

```powershell
python experiments/verify_shd_hardware_bundle.py
```

## Scope

PINES currently targets finite-horizon digital classification SNNs built from
current-based leaky integrate-and-fire neurons. The declared semantics cover
integration, threshold timing, reset, finite precision, rounding, saturation,
delivery delay, and deterministic or stochastic execution parameters. Analog
mismatch, online plasticity, continual learning, control, and energy claims are
outside the current scope.
