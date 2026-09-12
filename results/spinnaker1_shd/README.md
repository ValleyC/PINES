# SpiNNaker-1 SHD certificate evidence

`semantic_audit.json` contains the completed source-versus-emulator audit on
861 held-out inputs per condition. It covers five seeds before and after repair
and uses the paper-wide confidence allocation 0.05/(2 * 40).

`paired_predictions.npz` contains source and emulator class predictions for
the audit and disjoint canary inputs. It contains no physical predictions or
ground-truth labels. Array names identify the seed, split and model variant.

The original/repaired five-seed mean semantic bounds are approximately
33.7/20.8 percentage points. These are not accuracy measurements or physical
total bounds. The full physical canary campaign is still running. Its completed
analysis will be added here without substituting development captures for
the primary inputs.
