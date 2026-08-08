# Virtex-7 transport target

This RTL is a new, parameterized digital SRNN target for the declared execution
semantics. `lif_neuron.sv` selects integration, threshold timing, reset, signed
overflow, and fixed-point coefficients at synthesis time. The dense recurrent
core adds explicit synaptic and output delay stages.

The primary verification path is:

1. generate fixed-point traces from `transportcert.rtl_reference`;
2. run the same current/threshold sequence through cocotb;
3. compare every state and spike transition, not only the final prediction;
4. run bounded formal properties for overflow and reset invariants;
5. repeat against captured board traces with a hashed bitstream manifest.

The current checked-in environment has no HDL simulator, so software tests cover
the bit-accurate oracle while HDL execution remains an explicit hardware gate.
Before synthesis, the deployment generator must prove that every dense MAC fits
`STATE_BITS`; `dense_srnn_core.sv` intentionally labels the narrowing point.

