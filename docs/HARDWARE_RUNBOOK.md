# Hardware evaluation

## Common procedure

1. Freeze the model parameters, input IDs, execution mapping and run settings.
2. Check the target against the supplied software oracle and development traces.
3. Record the backend, tool/runtime versions, clock and logical timestep.
4. Run one primary execution per canary input, resetting model state between inputs.
5. Preserve captured predictions and available spikes, logits and raw traces.
6. Align predictions by sample ID, compute emulator-device disagreement and
   combine it with the separate source-emulator semantic bound.
7. Use ground truth only for post-capture accuracy evaluation. Keep repeated
   executions separate from the primary population sample.

## SpiNNaker-1 through EBRAINS

The supplied path uses PyNN and native SpiNNaker-1 neurons, not the SpiNNaker2
NIR converter. The [SHD guide](SPINNAKER1_EXPERIMENT.md) and
[DVS guide](SPINNAKER1_DVS_EXPERIMENT.md) specify weight orientation, neuron
parameters, delays, loading/reset profiles and host readouts.

The [retained measurements](../results/spinnaker1_retained/README.md) include
recorded logits, hidden spikes and paired predictions. Planned observations
without returned captures are unresolved in the reported population bound.
Complete-campaign analyzers and partial-capture accounting remain separate.

## FPGA mapping and Zynq-7000 captures

For SHD, use the parameters from `hardware/bundles/shd_floor_q8q16_v1/` with
the held-out inputs from `hardware/bundles/shd_virtex7_canary_v2/`. The earlier
training-pool input batch is separate from these 861 test-pool canaries.

For DVS, use `hardware/bundles/dvs_floor_q8q16_v1/`. It includes both
convolutional layers, recurrence, the integer readout, 104 audit recordings
and 160 canary recordings. Its README defines every memory layout and update.

The integer mapping is board-independent. Historical `virtex7` paths are
retained to keep scripts and frozen artifacts compatible. Actual supplied
Zynq-7000 CSVs are catalogued in [results/zynq_shd](../results/zynq_shd/README.md).
Capture metadata must identify the board actually used, not the historical
directory name. The legacy full SHD analyzer expects its `virtex7` manifest
value, so its backend field must be adapted before using it for a new board.

The checked-in RTL contains semantic unit-test cores, not a complete SHD/DVS
accelerator. Keep full-model RTL simulation, synthesis and physical-board
measurements distinct. Preserve tool versions, constraints, bitstream filename,
clock settings and raw board traces alongside each acquisition.

## Evaluate accuracy change

[hardware/evaluation](../hardware/evaluation/README.md) provides complete canary
ground truth, frozen source predictions and a NumPy-only delta A calculator.
It accepts class-prediction CSVs or NumPy captures without requiring a board
manifest. For DVS, first aggregate the four window logits with
`experiments/analyze_dvs_fpga_capture.py`.
