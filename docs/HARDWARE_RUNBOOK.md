# Physical backend runbook

## Shared preflight

1. Freeze model, NIR graph, sample IDs, semantics sidecar, adapter commit, and
   protocol version.
2. Execute the scalar/vector differential suite and backend emulator on the
   exact samples.
3. Record device serial, firmware, bitstream where applicable, conversion
   parameters, random seeds, and clock/timestep configuration.
4. Assign each primary input one independent hardware run pair ID.
5. Save raw spikes, predictions, and diagnostic state when available. Record the
   capture filename in the run manifest.

## SpiNNaker2

Pin `py-spinnaker2` and firmware before conversion. Use the official
`s2_nir.from_nir(graph, config=...)` route. Store the full conversion config and
represent every reset/discretization translation in the execution sidecar.
Repeated runs are a diagnostic artifact; the primary bound still uses independent
input/run pairs.

## Virtex-7

Generate a synthesis configuration from the exact semantics JSON. Verify every
neuron transition against the bit-accurate Python oracle and cocotb, then run
bounded reset/overflow properties. Store synthesis reports, tool versions,
constraints, firmware version, bitstream filename, board serial, raw UART/PCIe traces, and
clock settings.

The frozen SHD handoff is under
`hardware/bundles/shd_floor_q8q16_v1`. Start with its eight cycle-level smoke
inputs, then execute all 861 audit inputs in the recorded order. Run both the
unrepaired and repaired parameter sets. The input files contain no labels.
Compare physical predictions against the corresponding emulator prediction in
`golden_audit.npz`, not against the reference prediction. The latter is used by
PINES to compose reference--emulator and emulator--hardware disagreement.

## Stop conditions

Do not issue a physical certificate if sample ordering differs, pairs are reused,
the backend semantics description is missing, or firmware and bitstream versions
cannot be recovered. Such a run can diagnose a problem but is
not evidence for the primary claim.
