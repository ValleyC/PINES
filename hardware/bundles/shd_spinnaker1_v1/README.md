# Frozen SHD SpiNNaker-1 inputs and models

This bundle is used by the five-seed physical campaign. It contains the same
original SHD models as the FPGA handoff, plus the reset-to-value repairs selected
for the SHD manuscript study. It does not reuse the FPGA floor-quantization
repair as a substitute for reset repair.

- `models/<seed>/original.npz` is the frozen source checkpoint.
- `models/<seed>/reset_repaired.npz` is the label-free reset repair.
- `development_inputs.npz` contains eight repair-calibration inputs.
- `audit_inputs.npz` contains 861 held-out semantic-audit inputs.
- `canary_inputs.npz` contains 861 disjoint held-out physical-canary inputs.

Input archives contain `packed_spikes`, `sample_ids` and JSON `metadata`.
Use `numpy.unpackbits(packed_spikes, axis=-1, bitorder="little")[..., :700]`
to recover binary arrays in `[sample, timestep, channel]` order. Each input
has 50 timesteps and 700 channels. No ground-truth labels are included.

The hardware mapping uses reset-to-value `IF_curr_delta`, Euler-matched leak,
clocked bias, one-step recurrence, and a host linear readout. See
`docs/SPINNAKER1_EXPERIMENT.md` for the run command, evidence boundary and
frozen confidence allocation. These are input artifacts, not hardware results.

The original EBRAINS upload additionally contained copies of the execution
scripts. The repository retains those scripts in `experiments/` and `src/pines/`
without duplicating them in this compact bundle.
