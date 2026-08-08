# Physical backend runbook

## Shared preflight

1. Freeze model, NIR graph, sample IDs, semantics sidecar, adapter commit, and
   protocol version.
2. Execute the scalar/vector differential suite and backend emulator on the
   exact samples.
3. Record device serial, firmware, bitstream where applicable, conversion
   parameters, random seeds, and clock/timestep configuration.
4. Assign each primary input one independent hardware run pair ID.
5. Save raw spikes, predictions, and diagnostic state when available. Hash the
   capture before creating the immutable manifest.

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
constraints, firmware, bitstream hash, board serial, raw UART/PCIe traces, and
clock settings.

## Stop conditions

Do not issue a physical certificate if sample ordering differs, a capture hash
fails, pairs are reused, the backend semantics hash is missing, or firmware/
bitstream identity cannot be recovered. Such a run can diagnose a problem but is
not evidence for the primary claim.

