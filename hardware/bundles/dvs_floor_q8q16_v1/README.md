# Frozen DVS Gesture FPGA handoff

Use this package for FPGA/RTL, **not the SpiNNaker DVS mapping**. It contains
five frozen seeds, original and repaired parameters, inputs, and FPGA-specific
golden outputs. No retraining or original dataset download is required.
It is board-independent test data, not a bitstream or a complete accelerator.
The existing SHD-only RTL must be extended with both convolutional layers.

## Quick start

From the repository root (Python 3.11+, NumPy and PyTorch):

```sh
python -m pip install -e ".[torch]"
python experiments/verify_dvs_hardware_bundle.py
```

1. Start with seed `1701/unrepaired` and `common/development_input_spikes.mem`.
   These are two calibration recordings, not the held-out board experiment.
2. Compare states and spikes to `golden_development_traces.npz`. All traces are
   indexed `[sample,window,time,...]`. Membrane values are AFTER reset.
3. Compare every window's 11 final logits against
   `golden_development_window_logits.mem`. The host combines four windows.
4. After debugging, freeze the design and run `common/canary_input_spikes.mem`
   for all five seeds and both variants: 160 recordings per configuration,
   1,600 recording-level predictions in total. Do not use canary outputs to tune
   model parameters. Store repeats separately from this primary run.
5. Save the 11 signed integer final logits per window in a CSV with columns
   `sample_index,window_index,logit_0,...,logit_10`. Indices are zero-based.
   There are 640 rows per complete configuration. Preserve this raw file.
6. Convert and compare the output, for example:

```sh
python experiments/analyze_dvs_fpga_capture.py --seed 1701 --variant unrepaired --execution board --capture capture.csv --output hardware/captures/dvs_1701_unrepaired
```

The analyzer computes the host decision and emulator disagreement. It does not
pretend a partial capture is a complete experiment or calculate labeled accuracy.
For completed captures, the separate [hardware/evaluation](../../evaluation/README.md)
folder supplies all 160 aligned canary labels, five-seed source predictions,
and the command to calculate accuracy and delta A.
Also retain the bitstream, tool version, board model, clock, run settings and
whether the CSV came from RTL simulation or a physical board. Hardware clock
frequency is separate from the 60 logical SNN timesteps.

## Network and parameter layout

| Layer | Input -> output | Parameter shape |
|---|---|---|
| Conv1 | 2x32x32 -> 32x14x14 | 32x2x5x5 |
| Conv2 | 32x14x14 -> 64x6x6 | 64x32x3x3 |
| Hidden input | 2304 -> 256 | 256x2304 |
| Hidden recurrence | 256 -> 256 | 256x256 |
| Readout | 256 -> 11 | 11x256 |

Convolutions are **cross-correlations**, stride 2, no padding. Do not rotate
kernels. State flattening is channel, row, column (CHW), with column fastest.
Convolution weights are output-channel, input-channel, kernel-row, kernel-column
(OIHW), kernel-column fastest. Dense weights are destination-major/source-minor.
The `.npz` dense arrays are already `[destination,source]`, unlike the SHD
bundle's canonical NumPy matrices. Both bundles' `.mem` dense files are
destination-major. Do not transpose the DVS matrices again.

All weight files use signed 8-bit two's-complement integers with six fractional
bits. Bias, membrane, current, threshold and logits use signed 16-bit integers
with eight fractional bits. Both input and recurrent weights use the SAME
8-bit format. One scalar threshold (256 = 1.0) and one leak are shared by all
three spiking layers. The readout has no bias or spiking neuron.

Each `.mem` parameter line is one hexadecimal word. `parameters_int.npz` keeps
the array shapes. Five weight files and three bias files are provided for each
variant. No accumulator may wrap or saturate early during its dot product.

## Exact transition

The full contract is in `target_contract.json`. `sat16(a)` clamps to
[-32768,32767]. Compute one logical timestep in this order:

1. Conv1 receives the current binary input. Conv2 receives Conv1's spikes from
   THIS timestep. The hidden layer receives Conv2's current spikes and its own
   spikes from the PREVIOUS timestep. Initially all states/spikes/logits are 0.
2. For each neuron, accumulate the signed 8-bit weighted spikes in a signed
   32-bit accumulator. For hidden neurons, combine input and recurrent sums
   before saturation. Compute `I = sat16(4*weight_sum + bias_q16)`.
3. The stored leak is `floor(2^24/3) = 5592405 = 0x00555555`.
   Compute `v_pre = sat16(v + floor((I-v)*5592405 / 2^24))`.
   Use a signed 64-bit product and arithmetic right shift by 24. Negative
   division rounds DOWN, not toward zero.
4. Emit `q = (v_pre >= 256)` and set `v = sat16(v_pre - q*256)`.
   Reset is SUBTRACTIVE, not reset-to-zero. At most one spike per timestep.
5. Readout uses the new hidden spikes: `r = sat16(4*sum(q*W_out_q8))`, then
   `logits = sat16(logits+r)` EVERY timestep. Do not defer saturation to the end.
6. Repeat for 60 steps. Return all 11 final signed logits, reset ALL state, and
   run the next window. There are no additional logical delivery delays.

Physical pipeline cycles are fine, but must preserve this logical ordering.
The Q0.24 leak is an explicit hardware mapping, not exact division by 3.
Consequently these golden outputs and any later FPGA semantic bounds are
generated anew, not copied from the software-paper or SpiNNaker target.

## Inputs and classification

`common/` contains the same frozen events in `.npz` and `.mem` formats:

| Split | Recordings | Use |
|---|---:|---|
| development | 2 | Trace-level debugging |
| audit | 104 | Source-to-FPGA-emulator semantic audit |
| canary | 160 | Held-out physical-board comparison |

Each input line is one 2048-bit word (512 hex digits), one timestep. Bit
`p*1024+y*32+x` is input `(polarity=p,row=y,column=x)`. Bit 0 is the rightmost
hex digit's least significant bit. Sample `i`, window `w`, timestep `t` starts
at zero-based line `i*240+w*60+t`. Packed NumPy bytes use little-bit order.

One recording consists of FOUR independently reset windows. The host converts
each window's integer logits to real logits by dividing by 256, applies
softmax at temperature 0.5, then averages the four probability vectors. Choose
the lowest class index with maximum mean probability. Do not average raw
logits, take a majority vote, or count windows as independent test samples.
`analyze_dvs_fpga_capture.py` implements this postprocessing in float64.

## Golden files and reproducibility

Each seed has `golden_{development,audit,canary}.npz` and plain prediction CSVs.
The original source predictions are fixed for BOTH variants. The repaired
target is not compared against a repaired source. Per-variant golden `.mem`
logits are sample-major, then window-major, then class-major.

Each variant's `golden_development_traces.npz` includes all layer membranes,
spikes, accumulated logits, final window logits and recording predictions for
both development recordings. Traces describe logical timesteps, not FPGA cycles.

The package verifier checks memory order/encodings and reconstructs development
traces. Add `--full --device cuda` to regenerate all FPGA audit/canary logits.
The first development window is additionally checked against an independent
NumPy int64 implementation. Unit tests exercise CPU/GPU equality, floor rounding,
saturation, reset, window independence, packing and host aggregation.

Re-export from the included unchanged source models and inputs with:

```sh
python experiments/export_dvs_hardware_bundle.py --source-bundle hardware/bundles/dvs_floor_q8q16_v1 --output artifacts/dvs_fpga_reexport --device cuda
```

`verification.json` records the software checks actually performed. No physical
measurement or bitstream is included. Earlier DVS test performance informed
source-pipeline development, so these remain development-evidence inputs rather
than a newly sequestered benchmark. Dataset attribution is in `DATA_LICENSE.md`.
