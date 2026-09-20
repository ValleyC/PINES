# DVS Gesture on SpiNNaker-1

## Included results

The [retained physical archive](../results/spinnaker1_retained/README.md)
contains ten four-window recordings for seed 1701, before and after repair.
The planned matrix has 160 recordings per condition across five seeds.
Unreturned planned observations remain unresolved in the population bound.
`results/spinnaker1_dvs/` contains separate 104-recording semantic audits for
all ten seed/repair conditions and paired predictions for both frozen splits.

For FPGA work, use the [DVS FPGA handoff](../hardware/bundles/dvs_floor_q8q16_v1/README.md).
Its Q16-state arithmetic and subtractive reset differ from this mapping.

## Network and execution mapping

Both convolutional layers and the recurrent layer run on SpiNNaker-1. Their
32/64 convolutional channels correspond to 6,272/2,304 neurons, followed by
256 recurrent neurons. Only the linear readout and four-window aggregation
run on the host. The five original and floor-repaired models preserve the
manuscript architecture.

Weights are signed 8-bit floor-rounded values with six fractional bits.
Clocked biases are signed 16-bit values with eight fractional bits. Native
`IF_curr_delta` neurons use zero reset, post-integration thresholding, no
refractory interval and an Euler-matched exponential leak. State arithmetic
is native to the device, while the mapped emulator uses ideal float64 dynamics.
The pre-existing floor-target repair is evaluated under this mapped target.

Convolution projections preserve channel-height-width flattening and
cross-correlation kernel orientation. Linear projections transpose the
checkpoint's destination-source matrices into source-destination connections.
Positive and negative weights use separate receptor types.

Inputs start at physical step two. One-step projections align the first
convolution, second convolution and hidden outputs at steps three, four and
five. Clocked bias delays are one, two and three steps. Hidden recurrence has
one-step delay. Physical hidden bins five through 64 represent the 60 logical
model steps. Aligned reset reuse pads execution to 80 steps.

Four independently reset windows make one recording. For each window, the host
readout uses the quantized output weights, then softmax at temperature 0.5.
The mean of four probability vectors determines the class. Windows are not
independent classification samples.

## Runtime and input profile

The retained profile uses a 1 ms timestep, 100-fold wall-clock slowdown,
128 convolutional neurons per core, and 32 source/hidden neurons per core.
Byte-identical memory writes are limited to 262,144 bytes. Recorded packages
are sPyNNaker, SpiNNFrontEndCommon and SpiNNMan 7.4.2, PyNN 0.12.4 and NumPy 2.3.4.
The host compatibility shim preserves spike timing while handling input-array
replacement. Primary capture records hidden spikes with the `numpy-current`
reader. Full layer traces are available from calibration, not required for
recording-level classification.

The held-out input pool is divided into 104 semantic-audit recordings and
160 canary recordings using seed 20260914. Exported IDs use dataset row numbers
rather than filenames that encode labels. The DVS source pipeline is
reported as development evidence because test performance informed its design.

Each confidence term uses `0.05/(2*40)`. Across five seeds, original/repaired
semantic disagreement means are 71.5/18.7%, with upper bounds of 84.2/33.3 points.
The retained seed-1701 physical comparison uses its own semantic terms, not
these five-seed means. Even complete zero-disagreement audits at these sample
counts have a combined 11.355-point floor under the declared correction.

## Preparation and new execution

The original export command requires processed datasets and the original/final
repair checkpoints in the training workspace:

```sh
python experiments/export_spinnaker1_dvs_bundle.py --output artifacts/spinnaker1_dvs_bundle
```

The FPGA handoff includes corresponding frozen source models and event subsets,
but its golden outputs are not SpiNNaker emulator outputs. New physical runs
require a configured EBRAINS environment and an authorized hardware account.
`BUNDLE` below denotes the SpiNNaker export, not the FPGA directory.

Prepare the retained sample range with:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task dvs --bundle BUNDLE --inputs BUNDLE/canary_inputs.npz --seeds 1701 --start 0 --count 10 --reuse-reset --spike-reader numpy-current --output artifacts/dvs_canary.py --run-output dvs_canary
```

The remaining range for this seed is `--start 10 --count 150`. Other seeds use
`--start 0 --count 160`. Preserve all four windows and both model variants.
Use a separate working directory and `spynnaker.cfg` for per-run transfer
configuration. The supplied `hardware/configs/` files describe available
transfer profiles. Raw service logs and allocation handles remain private.

## Analysis

`prepare_spinnaker1_dvs_audit.py` computes the distinct mapped semantic terms.
`analyze_spinnaker1_dvs_matrix.py` analyzes a complete physical matrix.
`analyze_spinnaker1_repeats.py` summarizes separate repeated recordings without
increasing the primary sample count. Incomplete windows do not form a recording.

The published retained traces and prediction/bound arithmetic are reproducible
without hardware or original datasets:

```sh
python experiments/summarize_spinnaker1_retained.py --verify-only
```
