# Review artifact guide

## Included evidence

The [result inventory](../results/README.md) maps each manuscript table and
figure to its retained CSV, JSON or NumPy files. Software aggregates, supplied
RTL outputs and recorded physical observations are identified separately.

The repository includes the five-seed SHD/DVS original and repaired models used
by the FPGA handoffs, their frozen event subsets, integer memory images and
golden outputs. Original datasets and intermediate training trajectories are
not included. Hardware access is not needed to inspect the recorded results.

## Verify the published measurements

After the README installation steps, run from the repository root:

```sh
python experiments/verify_manuscript_results.py
python experiments/summarize_spinnaker1_retained.py --verify-only
python experiments/verify_repair_evidence.py
python experiments/verify_shd_hardware_bundle.py
python experiments/verify_dvs_hardware_bundle.py
```

The first command checks reported software values and retained SpiNNaker
Table V entries. The second reconstructs physical predictions from published
logits, aligns them with source/emulator outputs, and checks the reported bounds.
The bundle checks verify memory encodings, input ordering and development traces.
Use `--full --device cuda` with the DVS verifier to regenerate all included
audit/canary golden outputs when a suitable GPU is available.

Regenerate Figure 2 without rerunning experiments:

```sh
python experiments/plot_shd_hybrid_radius_sweep.py --output artifacts/review/figure2.pdf
```

## Rerun experiments

[Experiment entry points](../experiments/README.md) identify training,
semantics, family-analysis and repair scripts. Training requires the original
datasets and preprocessing described in those scripts. Long-running training,
full-family analysis and hardware acquisition are separate from the quick
verification commands above.

The operational scope and statistical assumptions are recorded in the
[research contract](RESEARCH_CONTRACT.md) and [evaluation protocol](EXPERIMENT_PROTOCOL.md).
Frozen config names and the original partition salt are retained because
changing them can change sample selection.

## Hardware and post-capture evaluation

The [hardware runbook](HARDWARE_RUNBOOK.md) links FPGA and SpiNNaker mappings.
New SpiNNaker execution requires an authorized EBRAINS account. New FPGA
execution requires the complete accelerator, a compatible board and its build
configuration. Checked-in RTL cores are transition-level reference modules.

The [FPGA evaluation package](../hardware/evaluation/README.md) contains aligned
ground-truth labels and original source predictions. Its evaluator computes
accuracy change from saved device predictions. DVS predictions aggregate four
windows into one recording before accuracy is evaluated.

The [Zynq inventory](../results/zynq_shd/README.md) lists the exact uploaded CSVs
and distinguishes their sample sets. The [SpiNNaker inventory](../results/spinnaker1_retained/README.md)
states the retained sample counts and treatment of unreturned planned pairs.

## Anonymous snapshot

This snapshot omits author and affiliation fields, account-specific screenshots,
private service job identifiers and personal machine paths. Published source
revisions use consistent opaque labels such as `review-revision-001` instead of
publicly searchable commit IDs. Numeric evidence and dataset attribution are
unchanged. Test fixtures use reserved example addresses, not author contacts.

Share the anonymous snapshot URL during review. The upstream Git hosting account
and its commit history are not anonymous and are not part of the review payload.
Local caches, raw service logs, credentials and cleanup backups are excluded
from the tracked snapshot.
