# DVS Gesture on SpiNNaker-1

## Scope and status

The new runner represents both convolutional layers and the recurrent layer on
the physical device. Only the linear readout and the existing four-window
aggregation run on the host. It preserves the trained architecture: 32 and 64
convolutional channels, 6,272 and 2,304 convolutional neurons, and 256 recurrent
neurons. The original and repaired checkpoints are the five-seed manuscript
checkpoints, not newly trained smaller networks.

The first development attempt mapped one window of seed 1701's original model
to 140 device cores. The service rejected the Java bulk-data transfer with HTTP
413 before neural execution. It produced no physical prediction. A local
configuration-file override used the home-file naming convention and was ignored.
An early programmatic setting was rejected before simulator setup. Inspection of
the installed loader identified the supported per-run file as `spynnaker.cfg`
without a leading dot. The current attempt selects Python transfer through that
file. That attempt completed a physical full-network window in 1,270.6 seconds,
including loading and capture. The full DVS canary campaign has not started.

Remote development bundle: `spinnaker1_dvs_development_v2`.
Completed attempt: `PINES_spinnaker1_runs/dvs_seed1701_original_development04_python`.
The separate `dvs_workspace_01` working directory isolates its report cleanup
and transfer configuration from the concurrently running SHD campaign.

A sequential comparison followed this attempt:
`PINES_spinnaker1_runs/dvs_seed1701_original_development05_direct`. It repeats
the same original model, development recording and window using
`hardware/configs/spinnaker1_python_direct_transfer.cfg`. The installed 7.4.1
loader supports direct per-region writes through
`Machine.disable_advanced_monitor_usage_for_data_in`. This changes loading, not
neuron dynamics or weights. The direct transfer was also rejected with HTTP 413
before execution. A subsequent development run, ending in `development06_chunked`,
uses contiguous memory writes of at most 256 KiB through the installed transceiver.
This changes only transfer granularity, preserving the data bytes and device
addresses. This run completed in 1,178.7 seconds. Its first-convolution,
second-convolution and hidden spike arrays and its logits exactly match the
completed Python-transfer capture. Its pre-shutdown diagnostics are also empty,
while shutdown retains router dump/reinjection warnings. The bounded direct
profile therefore has measured full-network loading evidence. Only one DVS
process runs at a time within the notebook server's memory limit.

The completed window uses seed 1701, the original model, the first development
recording and window zero. Hardware and emulator both predict class 3 for this
window. The maximum absolute logit difference is 0.59375.

| Layer | Physical spikes | Emulator spikes | Differing time-neuron cells |
|---|---:|---:|---:|
| First convolution | 42,319 | 42,312 | 87 |
| Second convolution | 5,115 | 5,116 | 59 |
| Recurrent hidden | 32 | 28 | 14 |

The recorded pre-shutdown packet query returned no late-spike entries. The full
log also contains router dump/reinjection warnings during shutdown provenance
collection. Retain those warnings with the capture rather than calling the run
warning-free. These traces establish a working full-network capture, not a
four-window classification result or a population certificate.

The remote archive `PINES_DVS_dev04_capture.zip` contains the raw and binned
spikes, logits, run configuration, development comparison and execution log.
It is preserved in the EBRAINS workspace. Local raw-capture import is pending.

The subsequent development run was
`PINES_spinnaker1_runs/dvs_seed1701_reset_reuse_development07`. It loads both the
original and repaired full models, executes two four-window calibration
recordings, and repeats the first window after each recording. The probe tests
whether PyNN simulation reset can reuse the loaded network without residual
states, stale inputs or packet problems. This is an execution-level reset,
distinct from a neuron's reset after firing. All layer traces and packet
diagnostics are retained. The exact-repeat test rejected this reuse profile,
as detailed below. The separate SHD campaign later stopped at its access quota.

The fresh-initialization NMPI batch profile completed all four windows of the
first development recording, with both original and repaired seed-1701 models.
It used sPyNNaker, SpiNNFrontEndCommon and SpiNNMan 7.4.2, PyNN 0.12.4 and NumPy
2.3.4. The original recording prediction is class 10 on both hardware and its
emulator. The repaired recording prediction is class 3 on both. These are
comparisons between executions, not a labeled accuracy measurement. Every
individual window prediction also agrees. Hardware and emulator trajectories
are not identical: the maximum logit difference across windows is 0.78125 for
the original and 6.625 for the repaired model. Recorded pre-shutdown late-spike
and message queries are empty, while the service reports retain router
dump/reinjection warnings.

The four window runtimes, including loading and capture, are 497.4, 486.5,
481.7 and 472.8 seconds. The local directory
`artifacts/spinnaker1_dvs_batch_development05/` retains the capture ZIP,
service reports and extracted per-layer spikes, logits, configuration and
recording-level development analysis. This completes one recording under two
conditions, not the planned 1,600-observation physical matrix. Fresh setup for
every window remains the selected execution profile. The declared v1 mapping
and existing checkpoints are fixed for the five-seed semantic audit.

The service timing databases identify the runtime bottleneck. Mean times across
these four windows are 311.3 seconds for application-data loading, 75.8 seconds
for data-specification generation, 16.1 seconds for buffer extraction, and 7.9
seconds for the application run. These measurements motivate reusing a loaded
network rather than extrapolating the fresh-allocation diagnostic into a quota
request. A 500,000-core-hour estimate based on assumed full-board charging was
withdrawn. No such allocation request was submitted.

## Mapping contract

Weights are floor-rounded to signed 8-bit values with six fractional bits.
Clocked biases use signed 16-bit values with eight fractional bits. The device
neuron is native `IF_curr_delta` with zero reset, post-integration thresholding,
no refractory period and an Euler-matched leak. Its state arithmetic is native
SpiNNaker arithmetic. The emulator is ideal float64, not the FPGA Q16-state
executor. Consequently, the earlier FPGA semantic bounds cannot be reused.

The existing floor-target repair is evaluated under this mapped target. It is
not described as having been retrained for native SpiNNaker arithmetic. The
physical comparison must measure whether its calibration transfers as well.

Convolution projections preserve PyTorch channel-height-width flattening and
cross-correlation kernel orientation. Linear projections transpose PyTorch's
destination-source storage to the source-destination connection convention.
Positive and negative weights use separate receptor types.

Input events start at physical step two. Successive one-step projections align
the first convolution, second convolution and hidden outputs at physical steps
three, four and five. Bias projections have delays one, two and three, so bias
updates start with each layer's first input. Hidden recurrence has one-step delay.
The 60 hidden-output bins are physical steps five through 64.

Four separately initialized windows make one recording-level observation.
Each window's spike counts feed the same floor-rounded readout weights, followed
by the checkpoint's mean-softmax aggregation with temperature 0.5. Individual
windows must not be counted as independent classifications in a binomial bound.

## Frozen input preparation

The bundle contains two training-pool repair-calibration inputs for development,
104 held-out semantic-audit inputs and the remaining 160 held-out canary inputs.
The two held-out sets are disjoint and cover the same test-input pool. Selection
uses seed 20260914. Dataset filenames can encode class annotations, so exported
identifiers use dataset row numbers rather than those filenames. Labels are not
exported. `dataset_indices` preserves the correspondence for later evaluation.

The reference DVS pipeline is still development evidence because test performance
informed its earlier design. Exporting new disjoint audit/canary subsets does
not make that history a preregistered confirmatory evaluation.

The planned physical matrix contains 160 inputs per seed and condition, five
seeds and original/repaired models, for 1,600 recording-condition observations.
Confidence allocation remains 0.05/(2 * 40), shared with the full paper matrix.
Its semantic audit is computed only after development fixes and mapping freeze.

The completed five-seed v1 audit is in `results/spinnaker1_dvs/`. Across 104
recordings per condition, original/repaired mean disagreement is 71.5/18.7
percent and the simultaneous semantic upper bound is 84.2/33.3 points. The
saved per-input predictions include the separate 160-recording canary split.
No labels were accessed. These are semantic terms for the mapped SpiNNaker
emulator, not measurements of physical accuracy or hardware conformance.

At these sample counts, even zero disagreements in both terms give bounds of
6.848 points for the semantic audit and 4.506 points for hardware conformance.
Their 11.355-point sum cannot accept the one-, two-, or five-point budgets.
Repeated windows are not extra independent recordings and cannot be used to
inflate the sample count. This experiment can measure physical transport and
repair, but these data alone cannot establish a tight five-point population
certificate under the declared simultaneous correction.

## Commands

Prepare models, input splits and development layer traces locally:

```sh
python experiments/export_spinnaker1_dvs_bundle.py --output artifacts/spinnaker1_dvs_development_v2
```

The exporter requires the archived original and final floor-repair checkpoints.
It computes only development predictions. It does not use audit/canary outcomes
to select the mapping. The complete kernel, signed-weight and aggregation tests
are in `tests/test_spinnaker1_dvs.py`.

Use an isolated working directory in the configured EBRAINS environment. Copy
`hardware/configs/spinnaker1_python_transfer.cfg` to `spynnaker.cfg` in that
directory, without a leading dot. This selects the supported Python transfer
path while retaining the account's existing home-directory connection settings.
The direct-transfer comparison uses `spinnaker1_python_direct_transfer.cfg` in
the same way. The runner's `--transfer-chunk-bytes` defaults to 262144 and records
the selected memory-transfer profile in its run configuration.

Run one complete-network development window first:

```sh
python experiments/run_spinnaker1_dvs.py --bundle BUNDLE --inputs BUNDLE/development_inputs.npz --output CAPTURE --seed 1701 --variants original --count 1 --windows 0 --record-layers
python experiments/analyze_spinnaker1_dvs_development.py --capture CAPTURE --bundle BUNDLE
```

The development analysis reports layer and logit differences for each window.
When all four window captures are present, it also compares the physical and
emulated recording predictions using the checkpoint's mean-softmax aggregation.
Partial windows do not enter this recording-level comparison. Neither diagnostic
produces a population certificate.

The allocation-reuse development probe uses only the two bundled calibration
recordings and writes separate diagnostic outputs:

```sh
python experiments/probe_spinnaker1_dvs_reset.py --bundle BUNDLE --output RESET_PROBE --seed 1701
```

The 7.4.1 reset-reuse development probe was rejected. Its exact repetition of the
first window reproduced every first-convolution spike, but the second convolution
and recurrent layer emitted no spikes after reset. The initial original-model
window had 5,115 second-convolution and 32 recurrent spikes. The repaired model
had 6,468 and 172, respectively. These disappeared in the repeat despite empty
reported late-spike counters. Consequently, this loading profile is not used for
primary certification. The incomplete probe and its exact-repeat traces remain
development evidence, not recording-level results.

A small physical two-layer diagnostic subsequently isolated a run-length effect.
At 67 steps, the initial downstream spikes at 4, 7 and 11 ms disappeared in both
reset repeats, while upstream spikes were unchanged. At 80 steps, both layers
reproduced their initial spike times exactly in both repeats. The installed
four-bit packet-colour period is 16 steps, making 80 the next aligned length.
This supports testing alignment padding after the observation window as a
network-reuse workaround. It does not yet establish full-model reset fidelity.
The first aligned run took 55.3 seconds and its repeats took 15.9 and 16.0 seconds
on this tiny network. These times are not full-DVS runtime estimates. Raw captures
and the completed report are retained locally under
`artifacts/spinnaker1_reset_alignment08/`.

The full-model calibration probe now accepts `--align-reset`. This pads execution
to the packet-colour period while retaining the original 60 observed hidden bins.
The default remains the previous unpadded profile. No primary canary observations
have been collected with the candidate aligned profile.

Build the full-model alignment test for the batch service:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task dvs-reset --bundle BUNDLE --seeds 1701 --align-reset --output artifacts/dvs_aligned_reset.py --run-output dvs_aligned_reset
```

This packages only the two development recordings and their emulator traces,
alongside both complete model variants. It does not package held-out canary inputs.
The aligned batch calibration attempt is in progress, not a completed primary
campaign. Preserve its capture and compare both repeated windows with their
initial physical execution before selecting the reuse profile.

The capture runner also has an optional `--reuse-reset` switch. It preserves
the existing per-recording output format and four-window aggregation, but loads
the networks once and resets before each subsequent window. Configuration and
per-window records distinguish this candidate profile from fresh allocation.
It is not the default, and its software orchestration tests are not evidence of
physical reset fidelity. The batch builder forwards this switch for DVS jobs.

A self-contained NMPI job can instead run each window with fresh initialization:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task dvs --bundle BUNDLE --inputs BUNDLE/development_inputs.npz --seeds 1701 --count 1 --windows 0 1 2 3 --record-layers --output artifacts/dvs_batch_development.py --run-output batch_dvs_development
```

This executes both original and repaired full networks. Use the submission flow
in `SPINNAKER1_EXPERIMENT.md` and preserve the resulting capture ZIP. Batch 7.4.2
and notebook 7.4.1 executions are separate development profiles.

The default runner executes all four windows and both model variants. It records
raw timestamps and layer spikes when requested. A recording-level prediction is
written only after all four windows complete. Preserve failed upload attempts,
logs and packet diagnostics alongside successful captures.

After the development mapping is fixed, prepare its distinct semantic audit:

```sh
python experiments/prepare_spinnaker1_dvs_audit.py --bundle BUNDLE --output AUDIT
```

This reconstructs the original float32 source executor for both conditions,
compares it with the mapped original/repaired emulator, and saves recording-level
predictions for the audit and canary splits. It never loads labels.

Analyze the five-seed physical captures once every complete recording is present:

```sh
python experiments/analyze_spinnaker1_dvs_matrix.py --captures SEED_1701 SEED_2718 SEED_3141 SEED_5772 SEED_8119 --audit AUDIT --output ANALYSIS.json
```

Incomplete runs produce progress counts only. Optional `--labels` points to the
original processed test archive for post-capture accuracy evaluation. The opaque
recording IDs are aligned through saved dataset row indices. Window-level
development results are never substituted for the full recording-level matrix.

## Repeated-run variability

Use separate capture directories for additional executions of the same fixed
inputs, checkpoints and execution configuration. These repeats describe device
variability, not additional independent deployment inputs:

```sh
python experiments/analyze_spinnaker1_repeats.py --task dvs --captures REPEAT_1 REPEAT_2 REPEAT_3 --output REPEATS.json
```

The report gives both the fraction of input-condition groups with any prediction
change and the fraction of executions that differ from the first execution.
Their denominators and individual prediction sequences are included. Only
complete four-window DVS classifications enter this analysis. Use `--task shd`
for the corresponding SHD captures. The primary confidence calculation does not
include these additional repeated executions.
