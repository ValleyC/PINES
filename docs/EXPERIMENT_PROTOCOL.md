# Frozen evaluation protocol v1

The machine-readable authority is `configs/protocol_v1.json`. Changes after the
first physical run require a new protocol version and must be reported.

## Tasks and partitions

- SHD recurrent SNN and DVS Gesture convolutional/recurrent SNN are primary.
- N-MNIST feedforward SNN is a transport-robust negative control.
- SSC begins only after every primary gate passes.
- The canonical training partition is deterministically hash-split into 80%
  training, 10% label-free repair calibration, and 10% unlabeled certificate
  audit. The canonical labeled test split remains sequestered evaluation.
- All five training seeds use identical sample IDs and semantic conditions.

Labels are inaccessible to repair and certification code paths. Repair choices
are made only on calibration IDs. Every selected repair is re-certified on audit
IDs that are checked disjoint at runtime. Labeled evaluation is read only after
the model, repair, certificate decision, and costs have been frozen.

## Conditions and comparisons

Single axes cover reset, integration, threshold timing, precision,
rounding/saturation, and one-step delay. Four high-risk combinations are frozen
in the protocol. Every method receives matched calibration examples, label
counts, optimization evaluations, wall-clock allowance, and reported hardware
runs.

Baselines are no repair, PTQ, global threshold scaling, per-platform QAT,
logit-only distillation, and fully supervised target retraining. Stage-2
per-neuron label-free calibration is primary; any few-shot stage reports its
nonzero label count.

## Outputs

Each cell records actual absolute accuracy change, total upper bound, slack,
static certified fraction, emulator/hardware disagreement, 1/2/5-point
accept/reject decisions, repair recovery, labels, runtime, hardware runs, and
all artifact hashes. Causal single-axis results precede combination results.

