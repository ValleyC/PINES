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
| Table V, SpiNNaker-1 SHD semantic term | `spinnaker1_shd/semantic_audit.json`, `spinnaker1_shd/paired_predictions.npz` |
| Table V, SpiNNaker-1 DVS semantic term | `spinnaker1_dvs/semantic_audit.json`, `spinnaker1_dvs/paired_predictions.npz` |
| Table V, retained SpiNNaker physical observations | `spinnaker1_retained/observations.csv`, `spinnaker1_retained/summary.json`, captured trace archives |
| Table V, Virtex-7 SHD semantic term | `virtex7_shd/semantic_audit.json`, `virtex7_shd/paired_predictions.npz` |
| Supplied Zynq-7000 SHD capture inventory | `zynq_shd/README.md`, four held-out canary CSVs and a separate earlier-batch CSV |

The finite-family source aggregates are retained beside each dataset. The SHD
continuous audit also retains its frozen screening summary because the final
all-input audit records it as upstream provenance.

The planned SpiNNaker-1 primary matrix covers 8,610 SHD and 1,600 DVS
predictions. Table V now uses the retained 1,000 SHD and 20 DVS physical
observations. Their ordered-prefix sampling is handled by counting unreturned
planned pairs as unresolved disagreements in the population bounds, with the
original confidence allocation. Empirical accuracy and disagreement describe
the recorded inputs. See `spinnaker1_retained/README.md` for sample counts,
per-seed aggregation and the reproducible calculation.
The older interrupted 75-input SHD run is separate from the current campaign.
The supplied FPGA simulation results are in `shd_rtl/` and do not replace
physical-board measurements. The separate supplied Zynq CSVs are catalogued in
[`zynq_shd/README.md`](zynq_shd/README.md). Historical `virtex7` paths below
identify the frozen integer mapping and are retained for compatibility.

The corresponding held-out SHD integer-FPGA audit is in
`virtex7_shd/semantic_audit.json`, with per-input predictions in the adjacent
`paired_predictions.npz`. Its five-seed mean semantic bounds are 85.3/25.4
points before/after repair. It uses the same source predictions and disjoint
test-input splits as SpiNNaker-1. Board inputs and expected integer outputs are
in `hardware/bundles/shd_virtex7_canary_v2/`. These software values fill only
Table V's semantic column, not its physical conformance or accuracy columns.

The DVS SpiNNaker-1 audit covers 104 held-out recordings for each of five seeds
and both original/repaired checkpoints. Its mean prediction-disagreement rates
are 71.5/18.7 percent and mean semantic upper bounds are 84.2/33.3 points.
`spinnaker1_dvs/paired_predictions.npz` retains the audit predictions and the
distinct 160-input canary predictions needed for the later physical comparison.
Each observation aggregates all four windows. The audit accesses no labels and
does not supply hardware-conformance or accuracy results. DVS remains development
evidence because earlier test accuracy informed its source pipeline.

Run the consistency check from the repository root:

```powershell
python experiments/verify_manuscript_results.py
```
