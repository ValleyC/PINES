# SHD manuscript evidence

The retained SHD artifacts support the manuscript's recurrent-model results.

- `software_matrix_*_float32` contains the five-seed, ten-target fixed-semantics
  matrix used in Table I.
- `static_family_memberwise_*` contains exact enumeration over the 16-member
  finite semantics family used by Table II.
- `hybrid_family_audit_summary.json` is the frozen screening aggregate recorded
  by the final continuous audit.
- `hybrid_family_full_audit_soundness_corrected_v1_summary.json` contains the
  five-seed continuous reset-family audit used by Table II.
- `hybrid_radius_sweep_*` and `shd_hybrid_radius_sweep.pdf` contain the matched
  continuous-width study used by Figure 2.
- `analysis_progression_*` contains the same-input affine, polygonal, and local
  branch progression used by Table III.
- `residual_counterexample_search_summary.json` records the independent
  falsification search associated with that development input.
- `repair_task_tuned_clean_v5_*` contains the matched repair comparison used by
  Table IV.
- `repair_task_tuned_family_grid_v5.json` contains the sampled post-repair
  family diagnostic discussed with Table IV.

All aggregate values are software evidence. Physical conformance measurements
are not included in these files.
