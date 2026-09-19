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
- `results/` contains the manuscript software results and separately labeled
  hardware-audit and RTL simulation evidence.
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

Table V reports physical SpiNNaker-1 and Virtex-7 results. The complete
SpiNNaker-1 capture matrix has been submitted through EBRAINS: 861 held-out SHD
inputs and 160 DVS recordings per condition, each covering five seeds and
original/repaired models. These are 8,610 SHD and 1,600 DVS primary predictions.
Physical execution is in progress. SHD recurrence and both DVS convolutions and
recurrence execute on the device, followed by host readouts. Each DVS
classification aggregates four windows. The five-seed DVS software audit in
`results/spinnaker1_dvs/` has mean semantic bounds of 84.2/33.3 points.

The aligned-reset loading profile passed SHD and DVS calibration repeats while
preserving the models and observation horizons. A current-segment spike reader
avoids rebuilding recording history after each reset. It matched the previous
reader on nine physical comparisons and reconstructed all 150 stored SHD/DVS
calibration traces exactly. Additional complete-input repeats are stored
separately for variability analysis and do not enlarge the primary sample count.
The retained primary captures now support the SpiNNaker table entries:
100 SHD inputs per seed across five seeds and ten DVS recordings for seed 1701.
Measured repair gains and full-population bounds are reported separately, with
unreturned planned pairs counted as unresolved disagreements. See
[`results/spinnaker1_retained/`](results/spinnaker1_retained/README.md).
See [`docs/SPINNAKER1_EXPERIMENT.md`](docs/SPINNAKER1_EXPERIMENT.md).
The distinct DVS mapping and its sample-complexity limit are described in
[`docs/SPINNAKER1_DVS_EXPERIMENT.md`](docs/SPINNAKER1_DVS_EXPERIMENT.md).

The supplied SHD RTL simulation outputs match all 8,610 frozen emulator class
predictions. These are recorded separately in
[`results/shd_rtl/`](results/shd_rtl/), not counted as physical measurements.

The SHD FPGA handoff is documented in
[`hardware/bundles/shd_floor_q8q16_v1/README.md`](hardware/bundles/shd_floor_q8q16_v1/README.md).
It contains all five frozen source models, their final certificate-directed
repairs for the floor-rounded fixed-point target, 861 unlabeled audit inputs,
integer memory images, emulator predictions, cycle-level smoke traces, and
plain experiment manifests. Verify it with:

```powershell
python experiments/verify_shd_hardware_bundle.py
```

The **DVS Gesture FPGA handoff** is in
[`hardware/bundles/dvs_floor_q8q16_v1/`](hardware/bundles/dvs_floor_q8q16_v1/README.md).
It includes five original/repaired models, FPGA memory images, two development
recordings with full layer traces, 104 audit recordings, and 160 held-out canary
recordings. Its Q8-weight/Q16-state/Q0.24-leak golden outputs are generated for
the FPGA mapping, not copied from SpiNNaker. The package requires a two-convolution
plus recurrent implementation and is not a bitstream. Verify it with
`python experiments/verify_dvs_hardware_bundle.py`; use
`experiments/analyze_dvs_fpga_capture.py` to compare captured window logits.

For the paper's held-out physical SHD comparison, use the new
[`Virtex-7 canary inputs`](hardware/bundles/shd_virtex7_canary_v2/README.md)
with the unchanged parameter files from that handoff. The old input batch and
its matching RTL CSVs remain separate simulation evidence. The new semantic
audit is in `results/virtex7_shd/`, alongside the distinct SpiNNaker-1 audit
in `results/spinnaker1_shd/`.

## Scope

PINES currently targets finite-horizon digital classification SNNs built from
current-based leaky integrate-and-fire neurons. The declared semantics cover
integration, threshold timing, reset, finite precision, rounding, saturation,
delivery delay, and deterministic or stochastic execution parameters. Analog
mismatch, online plasticity, continual learning, control, and energy claims are
outside the current scope.
