# Ground truth for completed FPGA runs

This folder supplies ground-truth labels and frozen source predictions to
calculate accuracy change after hardware capture. The original unlabeled input
and model packages are unchanged. No dataset download or source-model rerun is
needed. The evaluator uses only NumPy.

## Download the matching files

| Task | Files | Exact input package | Samples |
|---|---|---|---:|
| SHD | [CSV](shd_canary_labels.csv), [NumPy](shd_canary_labels.npz) | `hardware/bundles/shd_virtex7_canary_v2/` | 861 |
| DVS Gesture | [CSV](dvs_canary_labels.csv), [NumPy](dvs_canary_labels.npz) | `hardware/bundles/dvs_floor_q8q16_v1/` canary split | 160 |

Rows follow each package's `common/canary_samples.csv` exactly. `sample_index`
is the zero-based position in that hardware package, not the original dataset
row. Match `sample_id` when possible. These are **not** the first 861 or 160
labels of the original dataset. The older `shd_floor_q8q16_v1` training-pool
audit is a different 861-input set and must not use these labels.

CSV columns include `label` and `source_prediction_seed1701` through the five
seeds 1701, 2718, 3141, 5772, and 8119. The `label` is ground truth. Source and
golden emulator predictions are model outputs, not ground truth. Labels are
zero-based class IDs: 0-19 for SHD and 0-10 for DVS. Keep the same original
source prediction for both unrepaired and repaired targets.

## Calculate delta A

Evaluate each seed and repair state on the same sample IDs:

```text
source accuracy (%) = 100 * correct source predictions / sample count
hardware accuracy (%) = 100 * correct hardware predictions / sample count
delta A (points) = source accuracy - hardware accuracy
Table V entry = absolute value of delta A
```

Positive delta A means lost accuracy. Negative delta A means the hardware
model is more accurate on these inputs. For Table V, average the five
per-seed **absolute** changes separately for unrepaired and repaired targets.
Do not take the absolute value only after averaging signed changes.

For a SHD capture with `sample_ids` and `predictions` arrays:

```sh
python experiments/evaluate_fpga_delta_a.py --task shd --seed 1701 --variant unrepaired --predictions CAPTURES/1701/unrepaired/capture.npz --output hardware/captures/shd_1701_unrepaired_accuracy.json
```

Alternatively, provide a CSV with `sample_id,prediction` or
`sample_index,prediction`. The evaluator also accepts `hardware_prediction`
and `rtl_prediction` column names, as well as the uploaded `device_prediction`
format. Index-only files must follow the exact
canary input order above. It never substitutes golden predictions for captured
predictions. Keep results from RTL simulation and physical boards separate.

For DVS, first use the existing capture analyzer to combine all four windows:

```sh
python experiments/analyze_dvs_fpga_capture.py --seed 1701 --variant unrepaired --execution board --capture capture.csv --output hardware/captures/dvs_1701_unrepaired
python experiments/evaluate_fpga_delta_a.py --task dvs --seed 1701 --variant unrepaired --predictions hardware/captures/dvs_1701_unrepaired/predictions.npz --output hardware/captures/dvs_1701_unrepaired/accuracy.json
```

DVS has 160 **recording-level** labels, not 640 independent window labels.
The existing analyzer converts Q16 logits, applies softmax at temperature 0.5
per window, and averages four probability vectors before choosing the class.

The JSON reports both accuracies, signed `delta_a_pp`, `absolute_delta_a_pp`,
sample count and whether the canary is complete. It reports measurements on
the supplied inputs, not a population confidence bound. Labels are used for
this post-capture evaluation, not for label-free repair or certification.

The existing SHD physical-certificate analyzer also accepts
`--labels hardware/evaluation/shd_canary_labels.npz`.

## Reproduce the export

On the original workspace with the processed datasets:

```sh
python experiments/export_fpga_evaluation_labels.py
```

SHD labels are joined by the original sample IDs. DVS labels are joined through
the frozen dataset indices retained in the canary package. Source predictions
are copied from each seed's `golden_canary.npz`, without changing the models.

## Dataset attribution

Labels derive from Spiking Heidelberg Digits by Benjamin Cramer, Yannik
Stradmann, Johannes Schemmel and Friedemann Zenke, and IBM Research's DVS128
Gesture Dataset introduced by Arnon Amir et al. Dataset-derived labels retain
the Creative Commons Attribution 4.0 license, separately from the software.
See the existing [SHD attribution](../bundles/shd_virtex7_canary_v2/DATA_LICENSE.md)
and [DVS attribution](../bundles/dvs_floor_q8q16_v1/DATA_LICENSE.md).
