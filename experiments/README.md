# Manuscript experiment entry points

The experiment scripts are organized by the manuscript evidence they produce.
Large datasets, checkpoints, and raw reports are written below ignored `data/`
and `artifacts/` directories. Final aggregate evidence is written to `results/`.

## Fixed-target matrices

- SHD: `run_shd.py`, `run_shd_semantics.py`, `aggregate_shd.py`
- DVS Gesture: `run_dvs_gesture.py`, `run_dvs_semantics.py`,
  `aggregate_dvs_gesture.py`
- N-MNIST: `run_nmnist.py`, `run_nmnist_semantics.py`, `aggregate_nmnist.py`
- Metadata normalization: `correct_software_aggregate_float32.py`

## Family certificates

- Finite families: the `run_*_finite_family.py`, `run_shd_static_family.py`,
  and matching `aggregate_*` scripts
- Continuous SHD audit: `run_shd_hybrid_family_audit.py`,
  `run_shd_hybrid_family_full_audit.py`, validation scripts, and matching
  aggregators
- Contract-width aggregation: `aggregate_shd_hybrid_radius_sweep.py`
- Figure 2 from frozen results: `python experiments/plot_shd_hybrid_radius_sweep.py`
  (vector PDF, no experiment rerun). Use `--output` to select the manuscript's
  `figures/shd_hybrid_radius_sweep.pdf` destination.

## Analysis progression

The affine, guard-cut, polygon-branch, and counterexample-search runners feed
`aggregate_shd_analysis_progression.py`.

## Repair

The SHD and DVS repair runners produce matched method reports. Their aggregators
create the Table IV summaries. The repair-family-grid scripts produce the
sampled post-repair diagnostic.

## Verification

- `export_fpga_evaluation_labels.py` exports canary ground truth and frozen
  source predictions to `hardware/evaluation/`.
- `evaluate_fpga_delta_a.py` computes both accuracies and signed/absolute
  accuracy change from completed SHD or DVS recording predictions.
- `verify_manuscript_results.py` checks the rounded
  values used in the manuscript.
- `verify_repair_evidence.py` checks repair artifact identity and method
  comparisons.
- The hybrid-audit verifiers check row accounting, provenance, and soundness
  metadata.

## DVS FPGA handoff

- `export_dvs_hardware_bundle.py` exports frozen inputs/models, fixed-point
  memory images and target-specific golden outputs. It does not train a model.
- `verify_dvs_hardware_bundle.py` checks the images and reproduces development
  traces. `--full --device cuda` also regenerates all audit/canary logits.
- `analyze_dvs_fpga_capture.py` converts captured window logits into recording
  predictions and compares them with the FPGA emulator.

The ready-to-use package and run instructions are in
[`hardware/bundles/dvs_floor_q8q16_v1/`](../hardware/bundles/dvs_floor_q8q16_v1/README.md).
