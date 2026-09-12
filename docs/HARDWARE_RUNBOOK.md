# Physical backend runbook

## Shared preflight

1. Freeze the checkpoint, sample IDs, execution mapping, adapter version, and
   protocol. Include the NIR graph when that conversion route is used.
2. Execute the scalar/vector differential suite and backend emulator on the
   exact samples.
3. Record device serial, firmware, bitstream where applicable, conversion
   parameters, random seeds, and clock/timestep configuration.
4. Assign each primary input one independent hardware run pair ID.
5. Save raw spikes, predictions, and diagnostic state when available. Record the
   capture filename in the run manifest.

## SpiNNaker-1 through EBRAINS

The current physical path uses PyNN and native SpiNNaker-1 neurons, not the
SpiNNaker2 NIR converter. Preserve the explicit weight orientation, bias clock,
Euler-matched leak and spike-delivery alignment recorded in the run configuration.
SHD executes recurrence on the device with a host readout. DVS additionally
executes both convolutions on the device and retains host window aggregation.

Use [`SPINNAKER1_EXPERIMENT.md`](SPINNAKER1_EXPERIMENT.md) for the frozen SHD
campaign and [`SPINNAKER1_DVS_EXPERIMENT.md`](SPINNAKER1_DVS_EXPERIMENT.md) for
DVS mapping and development. These physical mappings have their own semantic
audits. Software bounds for the FPGA target are not substituted for them.

## Virtex-7

Generate a synthesis configuration from the exact semantics JSON. Verify every
neuron transition against the bit-accurate Python oracle and cocotb, then run
bounded reset/overflow properties. Store synthesis reports, tool versions,
constraints, firmware version, bitstream filename, board serial, raw UART/PCIe traces, and
clock settings.

The frozen SHD parameter handoff is under
`hardware/bundles/shd_floor_q8q16_v1`. Its earlier inputs and supplied batch CSVs
remain RTL simulation evidence. For the paper's held-out physical comparison,
use `hardware/bundles/shd_virtex7_canary_v2` with those same parameter files.
Run all 861 canary inputs for five seeds and both repair states. Compare against
the integer-emulator predictions in `golden_canary.npz`. The new disjoint
semantic audit and canary use the same held-out population as SpiNNaker-1.

The checked-in `rtl/` contains semantic unit-test cores, not a complete SHD or
DVS accelerator. Full-model simulation outputs do not establish physical-board
execution. The board implementation, build files, raw captures and configuration
remain part of the physical evidence needed from the hardware operator.

## Analysis

Align captured predictions by sample ID and keep one primary hardware execution
per input. Compute emulator-device disagreement on the canary, combine it with
the separate semantic term, then evaluate accuracy with labels. Preserve raw
traces, mapping details and implementation versions so readers can interpret the
result. Repeated executions describe variability and do not enlarge the primary
sample count. Incomplete captures provide progress and diagnostics only.
