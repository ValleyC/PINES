# Frozen SHD FPGA handoff

This bundle contains the inputs and parameters required to reproduce the PINES
physical conformance experiment for the floor-rounded Q8-weight/Q16-state
target. It is a deployment handoff, not a completed hardware result. No file in
this directory claims that a board run has occurred.

The modules in `rtl/` are semantic unit-test cores, not a complete SHD
accelerator. The SHD implementation must add the 700-input loader, 20-class
readout, and per-neuron Q0.24 leak path described below. The files in this bundle
are the common interface between that implementation and the Python emulator.

## Start here

1. Run `python experiments/verify_shd_hardware_bundle.py` from the repository
   root.
2. Begin with the first eight samples listed in `common/audit_samples.csv` and
   compare every timestep against `golden_smoke_traces.npz`.
3. After cycle-level agreement, execute all 861 inputs in exactly the recorded
   order using `common/audit_input_spikes.mem`.
4. Run each seed twice, once with `unrepaired/` and once with `repaired/`.
5. Save physical predictions with the matching sample IDs and a unique pair ID
   for every physical run. Preserve raw spikes and states separately.
6. Record the capture filename, bitstream filename, firmware version, tool
   version, and build configuration in the physical-run manifest.

The primary hardware comparison is against `unrepaired_emulator_predictions`
or `repaired_emulator_predictions` in each seed's `golden_audit.npz`. The
`reference_predictions` array is the fixed validated-source contract used to
compute the reference-to-emulator term of the PINES certificate.

## Shapes and formats

The network has 700 binary input channels, 128 recurrent hidden neurons, 20
readout classes, and 50 timesteps. Each sample occupies 50 consecutive lines in
`audit_input_spikes.mem`. The CSV index gives the first zero-based line for each
sample.

Each input line is one 700-bit hexadecimal word. Channel 0 is bit 0, which is
the least significant bit of the rightmost hexadecimal digit. The same bit
order is recorded in the compressed `audit_inputs.npz` file.

Weights are signed 8-bit fixed point with six fractional bits. State, current,
bias, threshold, reset value, and accumulated logits are signed 16-bit fixed
point with eight fractional bits. Each `.mem` parameter file contains one
two's-complement hexadecimal word per line.

The per-neuron forward-Euler leak is stored as an unsigned 32-bit word with 24
fractional bits. It is obtained by flooring `1/tau_mem` to Q0.24. This explicit
mapping supports the per-neuron leak values selected by repair and avoids an
unspecified floating-point division in the hardware contract.

The canonical NumPy matrices use `[source, destination]` order:

- input weights: `[700, 128]`
- recurrent weights: `[128, 128]`
- output weights: `[128, 20]`

The `.mem` matrices use destination-major, source-minor order. For example,
all 700 inputs to hidden neuron 0 appear first, followed by all inputs to hidden
neuron 1.

The weight integer scale is 64 and the state integer scale is 256. A weight
contribution must therefore be sign-extended and shifted left by two bits before
it is added to the state-domain accumulator. Accumulate at full precision, add
the Q16 bias, then apply signed 16-bit saturation. Do not simply truncate the
accumulator to its low 16 bits.

## Target transition

The exact target contract is stored in
`semantics/target_floor_q8q16.json`. For the included configuration:

1. Floor-quantize and saturate the weights and state-domain parameters.
2. Form the input and recurrent current from the current input spikes and the
   previous timestep's hidden spikes.
3. Convert the weighted sum to the Q16 state scale and saturate it.
4. Multiply `(current - membrane)` by the Q0.24 leak, arithmetically shift the
   signed product right by 24 bits, and add the result to the membrane. This
   realizes forward Euler integration with floor rounding.
5. Test the integrated state against the threshold.
6. Apply subtractive reset after the threshold test.
7. Accumulate the hidden-spike readout contribution into 20 Q16 logits.
8. Predict the lowest class index attaining the maximum final logit.

There is no synaptic or output delay in this target. An implementation with
truncation toward zero, wraparound, a different matrix order, or a different
tie rule is a different execution semantics and must not be compared against
these golden outputs as if it implemented this contract.

## Files per seed

- `manifest.json` lists the files, formats, seed, and upstream model and repair
  provenance.
- `golden_audit.npz` contains ordered predictions and final logits for all 861
  unlabeled inputs.
- `golden_smoke_traces.npz` contains per-timestep hidden membrane, spike, and
  logit traces for the first eight inputs.
- `unrepaired/original_model.npz` is the frozen trained source model.
- `repaired/original_model.npz` is the final certificate-directed repaired
  model chosen without audit labels.
- Each `deployed_model.npz` records the explicit Q16 state-parameter and Q0.24
  leak mapping used to produce the hardware memory images and golden outputs.
- Each variant's `parameters_int.npz` preserves shaped integer arrays, while
  its `.mem` files provide the hardware load order.

`manifest.json` at the bundle root lists the common files and every seed-level
manifest. Run the verifier after copying the bundle to another machine.

## Capture format

The primary board capture is a compressed NumPy file with three equal-length
one-dimensional arrays:

- `sample_ids`: the strings from `common/audit_samples.csv`
- `pair_ids`: unique identifiers for independent `(input, hardware run)` pairs
- `predictions`: integer class predictions in `[0, 19]`

Repeated executions used to characterize nondeterminism must be stored as a
separate diagnostic capture. They do not replace the independent primary
pairs.
