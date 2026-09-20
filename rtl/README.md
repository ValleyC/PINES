# FPGA semantic reference cores

The checked-in RTL implements parameterized execution-semantics test cores.
`lif_neuron.sv` selects integration, threshold timing, reset, signed overflow
and fixed-point coefficients. `dense_srnn_core.sv` adds recurrent processing
and explicit delay stages.

These modules are not the complete SHD or convolutional DVS accelerator.
The [hardware runbook](../docs/HARDWARE_RUNBOOK.md) links the full-model data
handoffs and supplied Zynq-7000 captures. Historical Virtex-7 mapping names
identify the original interface, not a claim about the board used for a result.

Verification proceeds from `pines.rtl_reference` traces to the same RTL current
sequences, comparing every state and spike. HDL execution requires an installed
simulator and cocotb. Bounded formal properties and captured board traces are
additional implementation checks, separate from Python unit tests.

Before synthesis, check that every dense accumulation fits `STATE_BITS`.
The narrowing point is explicitly marked in `dense_srnn_core.sv`.
