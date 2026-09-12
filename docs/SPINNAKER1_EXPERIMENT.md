# SpiNNaker-1 experiment

## Current campaign

The full SHD hardware campaign is running through EBRAINS: 861 held-out inputs,
five training seeds, and original/repaired models, giving 8,610 input-condition
observations. All inputs have been submitted in two disjoint ranges, 0--99 and
100--860. The batch 7.4.2 runtime uses the calibration-qualified aligned-reset
profile described below. All ten model conditions share each input execution.
Hidden recurrence executes on SpiNNaker-1, followed by the original linear
readout on the host. The remaining-input job uses current-segment spike reading
and captures a separate first-input repeat after its primary inputs finish.
The full physical bound awaits completion and analysis of both capture batches.

The submission manuscript is `transport_manuscript/`. The full five-seed DVS
matrix has also been submitted. Physical Virtex-7 captures remain outstanding.
The peer's matching
SHD predictions are RTL simulation evidence in `results/shd_rtl/`.
The DVS mapping and development progress are described in
[`SPINNAKER1_DVS_EXPERIMENT.md`](SPINNAKER1_DVS_EXPERIMENT.md).

### Earlier direct-access capture

The earlier fresh-allocation 7.4.1 run stopped after 75 complete inputs
(750 predictions) when the next allocation was rejected with `quota exceeded`.
Its captures and incomplete input 75 remain in
`PINES_spinnaker1_runs/shd_canary_five_seeds_v1/`, with a sibling log and the
`spinnaker1_ebrains_bundle_v8` execution bundle. These observations are separate
from the new 7.4.2 primary matrix. The NMPI batch allocation is separate from
the exhausted direct-access group.

## Development and frozen settings

Eight handcrafted cases cover integration, one- and two-step delays, inhibition,
bias, reset values, signed cancellation and recurrence. All expected spike
times matched. The largest one-step voltage residual was
0.00003128970165849898. The raw local capture is
`hardware/captures/spinnaker1/ebrains_20260912/probe_01/`.
It used sPyNNaker 7.4.1, PyNN 0.12.3 and SC&MP 4.0.0 on a 47-chip allocation.
This is handcrafted evidence, not full-model conformance.

Development uses eight frozen repair-calibration inputs, never held-out audit
or canary inputs. Initial executions exposed non-NumPy-indexable ragged tick
arrays and a stale send-buffer window cache in the 7.4.1 host software.
Compatibility fixes preserve exact spike times. Controlled source recordings
then showed that soft-reset executions could still report late packets and
change predictions. Fresh allocations avoided that reset-related behavior.

The eight-input, ten-condition validation finished with nine prediction
disagreements among 80 observations and no reported late-spike counters.
At ten-fold slowdown, the last input also produced source-buffer backpressure.
Repeating it at 100-fold slowdown removed those warnings and retained the same
three disagreeing predictions. The remaining warnings were a generic dense-source
warning and a deprecated configuration-version warning. Differences are measured,
not filtered out.

Frozen settings are a 1 ms simulation timestep, 100-fold wall-clock slowdown,
32 source and 32 hidden neurons per core, 2,048 incoming-buffer entries, and
two warm-up steps. Inputs start at physical step two. Hidden spikes at steps
three through 52 represent the model's 50 logical steps. Each input uses fresh
setup, execution, capture and shutdown.

## Model mapping

The target is `IF_curr_delta`: exponential integration, post-integration
`>=` thresholding, reset-to-value, no refractory interval and one-step
projections. Signed weights are separated into excitatory and inhibitory
connections while retaining the `[source, destination]` orientation.

For timestep `dt`, `cm=tau_m` gives unit membrane resistance. Projection
weights are `dt * W` because delta input is divided by `dt`. Bias uses a
clocked input channel so it starts with delayed inputs rather than before them.
`match_source_euler` sets `tau_m=-dt/log(1-dt/tau_source)`.
This matches the source leak in real arithmetic when
`0 < dt/tau_source < 1`. It neither equates reset rules nor eliminates device
fixed-point effects. `native_exponential` is implemented but not used in
this frozen campaign.

The emulator is ideal float64. Device arithmetic, weight realization, delivery
and execution variability contribute to measured conformance error. The installed
7.4.1 LIF, delta-input, delta-synapse, static-threshold and standard neuron
implementations were inspected for this mapping.

## Splits and confidence

SHD's held-out test set includes speakers absent from training. Both terms of
the physical certificate therefore concern that same held-out input pool.
The 861 canary inputs are selected without replacement with seed 20260912.
Another 861 disjoint semantic-audit inputs are selected with seed 20260913.
Execution files omit labels. The earlier training-pool software audit remains
separate and is not substituted into this bound.

`results/spinnaker1_shd/semantic_audit.json` records the completed semantic
audit for all ten conditions. Five-seed mean bounds are 33.7 points before repair
and 20.8 after repair. These bound decision disagreement, not measured accuracy
loss, and are not physical total bounds.

Confidence allocation is 0.05/(2 * 40) per term for the paper-wide 40
task/backend/seed/repair cells. Each condition uses one unique input/allocation
pair. Dependence across conditions is allowed by Bonferroni correction.
Representative independent inputs and allocation-level noise remain statistical
assumptions. Repeated development runs do not inflate the primary count.

All completed primary observations, including disagreements and warnings, are
retained. Interrupted allocations are recorded as incomplete. Resumption must
preserve completed observations and identify rerun attempts instead of silently
replacing outcomes.

## Reproduction

Use the official [EBRAINS access guide](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/QuickTry/)
and Jülich Lab with the EBRAINS-25.10 kernel. Run the provided
`SpiNNakerJupyterExamples/00.Setup/Setup.ipynb` to configure the service proxy.
Hardware access must be granted to the account. Credentials are not part of
the repository or execution bundle.

The original exporter requires the archived training and repair artifacts:

```sh
python experiments/export_spinnaker1_bundle.py --output artifacts/spinnaker1_ebrains_bundle_v8
python experiments/prepare_spinnaker1_audit.py --bundle artifacts/spinnaker1_ebrains_bundle_v8 --output artifacts/spinnaker1_physical_audit_v1
```

The compact frozen inputs and models are also retained in
`hardware/bundles/shd_spinnaker1_v1/`. With the repository scripts in the
configured hardware environment, run:

```sh
python experiments/run_spinnaker1_matrix.py --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/canary_inputs.npz --output hardware/captures/spinnaker1/shd_canary_v1 --count 861 --time-scale-factor 100
```

Each input folder records raw timestamps, binned spikes, logits, predictions,
machine description and packet diagnostics. Keep the full execution log because
shutdown can add warnings. Package versions and mapping parameters are recorded
without introducing cryptographic checks.

After completion, combine both terms and optionally supply labels only for
post-capture accuracy evaluation:

```sh
python experiments/analyze_spinnaker1_matrix.py --captures hardware/captures/spinnaker1/shd_canary_v1 --audit results/spinnaker1_shd --labels data/processed/shd_v1/test.npz --output results/spinnaker1_shd/physical_certificate.json
```

Incomplete captures return progress counts, not certificates. The analysis checks
captured emulator predictions against the frozen executor, counts primary pairs
once, and reports per-seed terms before aggregation.

## NMPI batch delivery

The NMPI Git-source option requires a public repository. For a private review
repository, build a self-contained Python job instead. It embeds the source,
models and input archive in a compressed payload and submits through the normal
authenticated NMPI client. No GitHub credential or repository visibility change
is needed. The generated job uses the same SHD runner and execution settings.

Start with development inputs to compare the batch runtime and captured spikes
with the notebook route:

```sh
python experiments/build_spinnaker1_nmpi_job.py --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/development_inputs.npz --output artifacts/shd_nmpi_development.py --seeds 1701 --count 1 --time-scale-factor 100
```

From an authenticated EBRAINS notebook with the existing allocated Collab:

```python
job = client.submit_job(
    source="artifacts/shd_nmpi_development.py", platform="SpiNNaker",
    collab_id=collab_id,
    command="run.py", tags=["PINES", "development"], wait=False)
```

The preinstalled batch environment observed on 2026-09-12 uses sPyNNaker,
SpiNNFrontEndCommon and SpiNNMan 7.4.2, PyNN 0.12.4 and NumPy 2.3.4 on Python
3.13.9. Requesting 7.4.1 failed in the service's version-switching script before
execution. Omitting that request allowed a one-input, two-condition SHD
development run to finish in 69.7 seconds. Both predictions matched the emulator.
This is development evidence, separate from the primary notebook campaign using
7.4.1. Inspect actual package versions and development traces before selecting a
batch profile. Preserve the generated `*_capture.zip`, which retains per-input
directories even when the service publishes individual files by basename.
Job IDs and private account details stay outside the anonymous artifact.

## Aligned reset reuse

The optional `--reuse-reset` runner keeps all selected SHD models loaded and
resets their state before replacing each input. It pads the physical run from
55 to 64 steps to align the four-bit packet-colour counter, while retaining
exactly the same 50 hidden-output bins. This is a separate execution profile,
not a continuation of the interrupted fresh-allocation 7.4.1 campaign.

The source-update compatibility shim now covers both observed 7.4.1 and batch
7.4.2 runtimes. The 7.4.2 DVS calibration exposed the same ragged-array indexing
failure when replacing inputs. The shim changes host containers and buffer
refill bookkeeping, not spike times or device binaries.

The five-seed SHD reset calibration uses the existing eight development inputs
and then repeats the first. Its `repeat_first/` directory is separate from
primary input folders and never increases the population sample count:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task shd --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/development_inputs.npz --count 8 --reuse-reset --repeat-first --output artifacts/shd_reset_calibration.py --run-output shd_reset_calibration
```

The batch 7.4.2 calibration completed all eight inputs and the separate repeat.
Every one of the ten models reproduced its first-run binned spikes and logits
exactly. Hardware and emulator disagreed on nine of 80 calibration predictions.
The saved per-run diagnostic queries contain no late-spike entries or messages,
while the full service log retains dense-source and configuration warnings.
The initial run took 76.6 seconds. Subsequent inputs averaged 23.6 seconds, with
a range of 22.1--26.9 seconds. The raw capture and service reports are retained
locally under `artifacts/spinnaker1_shd_aligned_reset11/`.

The primary campaign now uses this fixed profile in bounded batches. The first
batch is:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task shd --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/canary_inputs.npz --start 0 --count 100 --reuse-reset --output artifacts/shd_canary_aligned_v1_0000_0099.py --run-output shd_canary_aligned_v1_0000_0099
```

It has started physical execution. The remaining 761 inputs have also been
submitted as one job, with the same models, sampling and execution settings:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task shd --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/canary_inputs.npz --start 100 --count 761 --reuse-reset --repeat-first --spike-reader numpy-current --output artifacts/shd_canary_aligned_v1_0100_0860.py --run-output shd_canary_aligned_v1_0100_0860
```

The second job reads current-segment spikes through the physically compared
array API instead of rebuilding Neo history. It adds one separate first-input
repeat after completing its primary inputs. The two primary ranges are disjoint
and together cover all 861 inputs. Pass both extracted capture directories to
the matrix analyzer after they finish. The repeat analyzer reads the additional
`repeat_first/` capture without adding it to the certificate sample count.
Calibration observations and the older 75-input campaign are excluded from this
new primary analysis. Labels remain reserved for post-capture evaluation.

## Collecting the running matrix

`experiments/collect_spinnaker1_campaign.py` accepts an authenticated NMPI client
and the handles of jobs already submitted. It polls job details every five
minutes, retains logs and capture archives once a job ends, and analyzes SHD or DVS after all its
jobs end. It neither submits new jobs nor loads ground-truth labels. In the
experiment notebook, run it in a background thread so the kernel stays usable:

```python
from threading import Thread
from collect_spinnaker1_campaign import collect_campaign

# Each entry identifies an existing submission, not a new experiment.
# jobs = [{"job": job_handle, "task": "shd", "name": "shd_batch"}, ...]
worker = Thread(target=collect_campaign, args=(client, jobs, "collected", {
    "shd": "results/spinnaker1_shd", "dvs": "results/spinnaker1_dvs"}), daemon=True)
worker.start()
```

The deployed service returned 500/502 errors for the `with_log=False` request
form while the original `with_log=True` request still returned running jobs.
The collector uses the working form and keeps only the latest progress line
between polls. Observation failures do not trigger resubmission.

`progress.json` records collection status. Each job directory retains its capture
ZIP, extracted observations, service log and available report archive. Per-task
label-free and repeat reports are separate. An incomplete capture remains an
incomplete analysis even when the service job has ended. Final scientific reports
include sample ranges, reset/loading profiles, reader choice and package versions
from the capture configurations. Private service logs and account-related job
bookkeeping remain outside the anonymous repository.

Review the final report archive as well as per-input diagnostics. Shutdown can
retrieve provenance that was not available to an earlier per-input query. The
following command summarizes warning categories without including account paths
or service identifiers in its output:

```sh
python experiments/summarize_spinnaker1_reports.py --reports BATCH_1/reports.zip BATCH_2/reports.zip --output EXECUTION_LOG_SUMMARY.json
```

Counts refer to log entries, not unique incidents or lost packets. In particular,
router counters can recur across resets and packet reinjection. Keep every
primary prediction in the conformance analysis and retain the raw logs for
interpretation. Empty per-input diagnostics alone do not establish a warning-free
hardware run.
