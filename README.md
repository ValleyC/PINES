# PINES

**Prediction Invariance across Neuromorphic Execution Semantics for Spiking Neural Networks**

Anonymous review artifact. PINES analyzes prediction preservation when a trained
spiking neural network moves between execution semantics. It combines per-input
family analysis, statistical disagreement bounds, measured hardware conformance,
and label-free repair.

## Quick start

Python 3.11 or newer is required. From the repository root:

```sh
python -m pip install -e ".[dev,torch]"
python -m pytest
python experiments/verify_manuscript_results.py
python experiments/summarize_spinnaker1_retained.py --verify-only
```

These checks use the included artifacts. They do not require a hardware account
or the original training datasets. CUDA tests run when a supported GPU is
available. A small end-to-end software example is also provided:

```sh
pines reproduce --config configs/toy_study.json --output-dir artifacts/local-toy
```

## Manuscript evidence

| Evidence | Location |
|---|---|
| Fixed-target results, family certificates and repair comparisons | [Result inventory](results/README.md), Tables I-IV |
| Continuous-contract sensitivity | [Figure 2 data](results/shd_v1/hybrid_radius_sweep_summary.json) and [vector figure](results/shd_v1/shd_hybrid_radius_sweep.pdf) |
| SpiNNaker-1 physical observations and bounds | [Retained measurements](results/spinnaker1_retained/README.md), Table V |
| Supplied Zynq-7000 captures | [SHD capture inventory](results/zynq_shd/README.md) |
| Separate SHD RTL simulation results | [RTL evidence](results/shd_rtl/README.md) |

The retained SpiNNaker results contain 100 SHD inputs for each of five seeds
and ten DVS recordings for seed 1701, before and after repair. Their population
bounds retain the planned sample counts and count unreturned pairs as unresolved.
The Zynq capture inventory distinguishes held-out canaries from an earlier input
batch. Measured values, model parameters, sample IDs and numeric arrays are
preserved in this review snapshot.

The [artifact guide](docs/ARTIFACT_GUIDE.md) explains verification, reproduction
requirements, hardware access and the correspondence between results and code.

## Hardware packages

| Task | Frozen package | Evaluation |
|---|---|---|
| SHD FPGA | [Parameters and development traces](hardware/bundles/shd_floor_q8q16_v1/README.md), then [held-out canary inputs](hardware/bundles/shd_virtex7_canary_v2/README.md) | 861 canary inputs per seed and variant |
| DVS FPGA | [Convolutional/recurrent FPGA handoff](hardware/bundles/dvs_floor_q8q16_v1/README.md) | 104 audit and 160 canary recordings, four windows each |
| SHD SpiNNaker-1 | [Frozen model/input bundle](hardware/bundles/shd_spinnaker1_v1/README.md) | [Mapping and execution guide](docs/SPINNAKER1_EXPERIMENT.md) |
| DVS SpiNNaker-1 | [Model preparation and mapping](docs/SPINNAKER1_DVS_EXPERIMENT.md) | Separate mapped emulator and physical execution |

The FPGA bundles provide memory images, model parameters and golden outputs,
not a complete board bitstream. Historical `virtex7` paths identify the integer
mapping and remain stable for reproducibility. The supplied physical board
captures are labeled Zynq-7000.

[Ground-truth labels, original source predictions and the delta A calculator](hardware/evaluation/README.md)
are provided separately for completed FPGA canary runs. No retraining is needed
to evaluate saved predictions.

## Repository layout

- `src/pines/`: execution semantics, certification, repair and adapters.
- `configs/`: model-independent protocols and experiment settings.
- `experiments/`: training, evaluation, plotting and verification entry points.
- `results/`: retained numeric evidence and physical capture inventories.
- `hardware/`: frozen handoffs, post-capture labels and run specifications.
- `rtl/`: parameterized semantic test cores and verification harnesses.
- `schemas/`: report and execution-contract formats.
- `tests/`: software, differential, statistics and data-layout tests.

## Scope and licensing

The manuscript evaluates finite-horizon digital classification with current-based
leaky integrate-and-fire neurons. Analog mismatch, online plasticity, continual
learning, control and energy-efficiency claims are outside this evaluation.

Software is released under the [MIT license](LICENSE). Dataset-derived inputs
and labels retain their source licenses and attribution in each hardware bundle.
See [contribution guidelines](CONTRIBUTING.md) for changes to the review artifact.
