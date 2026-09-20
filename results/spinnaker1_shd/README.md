# SpiNNaker-1 SHD certificate evidence

`semantic_audit.json` contains the completed source-versus-emulator audit on
861 held-out inputs per condition. It covers five seeds before and after repair
and uses the paper-wide confidence allocation 0.05/(2 * 40).

`paired_predictions.npz` contains source and emulator class predictions for
the audit and disjoint canary inputs. It contains no physical predictions or
ground-truth labels. Array names identify the seed, split and model variant.

The original/repaired five-seed mean semantic bounds are approximately
33.7/20.8 percentage points. These are not accuracy measurements or physical
total bounds. The recorded primary observations and their unresolved-pair
physical bounds are in [spinnaker1_retained](../spinnaker1_retained/README.md).
That archive states the exact returned sample count for each condition.
