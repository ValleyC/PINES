# Virtex-7 transport target

This RTL is a parameterized semantic unit-test target. `lif_neuron.sv` selects
integration, threshold timing, reset, signed overflow, and fixed-point
coefficients at synthesis time. The dense recurrent core adds explicit synaptic
and output delay stages. It is not the complete 700-input, 20-output SHD
accelerator described by `hardware/bundles/shd_floor_q8q16_v1/README.md`.

The primary verification path is:

1. generate fixed-point traces from `pines.rtl_reference`;
2. run the same current/threshold sequence through cocotb;
3. compare every state and spike transition, not only the final prediction;
4. run bounded formal properties for overflow and reset invariants;
5. repeat against captured board traces while recording the bitstream filename.

The current checked-in environment has no HDL simulator, so software tests cover
the bit-accurate oracle while HDL execution remains an explicit hardware gate.
Before synthesis, the deployment generator must prove that every dense MAC fits
`STATE_BITS`; `dense_srnn_core.sv` intentionally labels the narrowing point.
