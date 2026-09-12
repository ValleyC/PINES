# Manuscript result artifacts

This directory contains the manuscript software evidence and separately labeled
RTL simulation and hardware-audit evidence. JSON summaries preserve protocol details, provenance
source filenames and full-precision statistics. CSV files contain the corresponding
rows. Reported manuscript values are rounded from these artifacts.

## Evidence map

| Manuscript item | Result artifacts |
|---|---|
| Table I, fixed-target SHD | `shd_v1/software_matrix_rows_float32.csv`, `shd_v1/software_matrix_summary_float32.json` |
| Table I, fixed-target DVS Gesture | `dvs_gesture_v3/software_matrix_rows.csv`, `dvs_gesture_v3/software_matrix_summary.json` |
| Table I, fixed-target N-MNIST | `nmnist_v1/software_matrix_rows_float32.csv`, `nmnist_v1/software_matrix_summary_float32.json` |
| Figure 2, continuous-width sensitivity | `shd_v1/hybrid_radius_sweep_rows.csv`, `shd_v1/hybrid_radius_sweep_summary.json`, `shd_v1/shd_hybrid_radius_sweep.pdf` |
| Table II, family certificates | `family_certificates/software_family_rows.csv`, `family_certificates/software_family_summary.json` |
| Table II, continuous SHD source audit | `shd_v1/hybrid_family_full_audit_soundness_corrected_v1_summary.json` |
| Table III, analysis progression | `shd_v1/analysis_progression_rows.csv`, `shd_v1/analysis_progression_summary.json` |
| Table IV, SHD repair | `shd_v1/repair_task_tuned_clean_v5_rows.csv`, `shd_v1/repair_task_tuned_clean_v5_summary.json` |
| Table IV, DVS Gesture repair | `dvs_gesture_v3/repair_floor_task_tuned_clean_v5_rows.csv`, `dvs_gesture_v3/repair_floor_task_tuned_clean_v5_summary.json` |
| Repair family diagnostic | `shd_v1/repair_task_tuned_family_grid_v5.json` |
| Audit sample requirements | `sample_complexity/zero_disagreement_rows.csv`, `sample_complexity/zero_disagreement_summary.json` |

The finite-family source aggregates are retained beside each dataset. The SHD
continuous audit also retains its frozen screening summary because the final
all-input audit records it as upstream provenance.

The full SHD SpiNNaker-1 canary campaign is running. The semantic term for its
held-out population is in `spinnaker1_shd/semantic_audit.json`. It is not a
physical total bound. Table V's hardware-dependent values remain unreported
until complete physical captures are available. The supplied FPGA simulation
results are in `shd_rtl/` and do not replace physical Virtex-7 measurements.

Run the consistency check from the repository root:

```powershell
python experiments/verify_manuscript_results.py
```
