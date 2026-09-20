# Supplied Zynq-7000 SHD captures

Raw prediction CSVs are preserved without changing their values. The folder
separates held-out canary recordings from an earlier index-only input batch.

## Held-out canary CSVs

| Seed | Variant | File | Samples |
|---|---|---|---:|
| 2718 | unrepaired | [CSV](seed2718_unrepaired_canary_hw.csv) | 861 |
| 2718 | repaired | [CSV](seed2718_repaired_canary_hw.csv) | 861 |
| 3141 | unrepaired | [CSV](seed3141_unrepaired_canary_hw.csv) | 861 |
| 3141 | repaired | [CSV](seed3141_repaired_canary_hw.csv) | 861 |

These files contain explicit sample IDs matching
`hardware/bundles/shd_virtex7_canary_v2/common/canary_samples.csv`. Each file's
861 device predictions match its corresponding frozen integer-emulator output.
They cover four of the ten planned seed/variant conditions. Board build files
and complete five-seed acquisition metadata are separate from this CSV inventory.

Aligned [ground truth and source predictions](../../hardware/evaluation/README.md)
support post-capture accuracy evaluation, for example:

```sh
python experiments/evaluate_fpga_delta_a.py --task shd --seed 2718 --variant repaired --predictions results/zynq_shd/seed2718_repaired_canary_hw.csv
```

The CSV field `device_prediction` is accepted directly. The evaluation command
does not alter these files or declare a complete physical certificate.

## Earlier input batch

[`earlier_input_batch/seed1701_unrepaired_hw.csv`](earlier_input_batch/seed1701_unrepaired_hw.csv)
contains 861 index-only predictions. Its values exactly match the unrepaired
emulator output for the older training-pool batch in
`hardware/bundles/shd_floor_q8q16_v1/`, rather than the held-out canary output.
The file has no sample-ID column. Keep it separate and do not apply the canary
labels or merge it into the held-out physical matrix.

Terminal screenshots are excluded from the review snapshot. The numeric CSVs
above are the retained machine-readable evidence.
