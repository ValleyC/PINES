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
- Contract-width figure: `aggregate_shd_hybrid_radius_sweep.py`

## Analysis progression

The affine, guard-cut, polygon-branch, and counterexample-search runners feed
`aggregate_shd_analysis_progression.py`.

## Repair

The SHD and DVS repair runners produce matched method reports. Their aggregators
create the Table IV summaries. The repair-family-grid scripts produce the
sampled post-repair diagnostic.

## Verification

- `verify_manuscript_results.py` checks the rounded
  values used in the manuscript.
- `verify_repair_evidence.py` checks repair artifact identity and method
  comparisons.
- The hybrid-audit verifiers check row accounting, provenance, and soundness
  metadata.
