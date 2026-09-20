# Evaluation protocol

`configs/protocol_v1.json` records the original study design. Per-study configs,
result summaries and frozen sample IDs specify the executed experiments.
Existing partition salts, seeds and filenames are retained unchanged.

## Tasks and partitions

- SHD: 700 input channels and 128 recurrent neurons, 50 binary steps.
- DVS Gesture: two convolutional layers and 256 recurrent neurons, four
  60-step windows per recording.
- N-MNIST: 128 feedforward neurons, 30 steps, used as a transport-robust control.
- Five training seeds: 1701, 2718, 3141, 5772 and 8119.

Software training/repair/audit splits follow the frozen 80/10/10 allocation.
Per-seed audit sizes are 861/104/6,005 for SHD/DVS/N-MNIST. SHD and DVS repair
sets contain 800/97 unlabeled inputs and remain disjoint from their audit sets.
N-MNIST is not repaired. Physical semantic audits and canaries use separate
held-out test-pool splits specified by each backend bundle.

SHD/N-MNIST test labels are reserved for evaluation. DVS and repair comparisons
are marked as development evidence in the manuscript. DVS test performance
informed source-pipeline development. Labels are not used by the label-free
repair or certificate calculation. Supervised baselines report their label use.

## Conditions and comparisons

Ten fixed targets vary integration, threshold timing, reset, finite precision,
rounding/saturation and delay individually or jointly. Finite-family analysis
covers 16 combinations. The continuous SHD study varies timestep and threshold
under reset-to-value, with fixed proof budgets and explicit unresolved outcomes.

Repair baselines include no repair/PTQ, global threshold scaling, logit-only
distillation, target QAT and few-shot training from random initialization.
Gradient methods use 280 updates with the samples, labels and trainable
parameters recorded in their retained summaries.

## Outputs and interpretation

Each experiment reports accuracy change, prediction disagreement and the
applicable upper bound. Family studies additionally report certified, falsified
and unresolved inputs. Repair studies report recovery, post-repair bounds,
labels and optimization cost. Physical studies add measured emulator-device
mismatch and retain captured observations separately from emulator outputs.

The decision budgets are one, two and five percentage points. Confidence
allocation is specified for each experiment, including `0.05/(2*40)` per
physical term. In the retained partial SpiNNaker batches, unreturned planned
pairs count as unresolved disagreements. Repeats characterize variability
and do not enlarge the primary population sample.
