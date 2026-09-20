# SHD on SpiNNaker-1

## Included results

The [retained physical archive](../results/spinnaker1_retained/README.md)
contains 100 held-out inputs per seed across five seeds and original/repaired
models, totaling 1,000 input-condition predictions. The planned canary has
861 inputs per condition. Returned inputs are an ordered prefix of that frozen
set, so its population bound counts unreturned pairs as unresolved.

The complete 861-input source-emulator audits and paired predictions are in
`results/spinnaker1_shd/`. Five-seed semantic bounds are 33.7/20.8 points
before/after repair. Complete-campaign analysis and retained-prefix accounting
are separate entry points. No live job status is implied by this snapshot.

## Execution mapping

The physical neuron is native `IF_curr_delta`, with post-integration `>=`
thresholding, reset-to-value, no refractory interval and one-step projections.
Positive and negative weights use separate excitatory/inhibitory connections
with `[source, destination]` orientation. Recurrence executes on the device.
The original linear readout on the host sums recorded hidden spikes.

For timestep `dt`, `cm=tau_m` gives unit membrane resistance and projection
weights are `dt * W`. A clocked input channel applies bias with the input.
The mapping `tau_m=-dt/log(1-dt/tau_source)` matches the source Euler leak in
real arithmetic for `0 < dt/tau_source < 1`. The mapped float64 emulator does
not model native fixed-point arithmetic. These residual differences enter
the measured emulator-device term.

The retained execution profile uses a 1 ms simulation timestep, 100-fold
wall-clock slowdown, 32 source/hidden neurons per core and 2,048 incoming-buffer
entries. Inputs begin at physical step two. Hidden bins three through 52
represent the 50 logical model steps. Aligned reset reuse pads each run to
64 physical steps without changing those observation bins.

Recorded versions are sPyNNaker, SpiNNFrontEndCommon and SpiNNMan 7.4.2,
PyNN 0.12.4 and NumPy 2.3.4. The host compatibility shim handles ragged tick
arrays and replaced-buffer cache invalidation without changing spike times or
device arithmetic. Exact settings are retained in each execution profile in
`results/spinnaker1_retained/summary.json`.

## Data and confidence

[The frozen bundle](../hardware/bundles/shd_spinnaker1_v1/README.md) includes
original and reset-repaired models, eight calibration inputs, 861 semantic-audit
inputs and 861 disjoint canary inputs. Physical audit/canary inputs come from
the held-out test pool, selected with seeds 20260913/20260912 respectively.

Confidence allocation is `0.05/(2*40)` per term. Conditions may be dependent
under Bonferroni correction. Representative independent inputs and
execution-level hardware noise remain statistical assumptions. Repeated
executions are diagnostics, not additional primary samples. All recorded
primary predictions are retained, including disagreements and warnings.

## New physical execution

Use an authorized EBRAINS account and the service's notebook setup to configure
its proxy. Account credentials, allocation handles and raw service logs are
not included. Hardware execution is separate from the local result-verification
commands in the repository README.

A self-contained NMPI job for the retained sample range can be prepared with:

```sh
python experiments/build_spinnaker1_nmpi_job.py --task shd --bundle hardware/bundles/shd_spinnaker1_v1 --inputs hardware/bundles/shd_spinnaker1_v1/canary_inputs.npz --start 0 --count 100 --reuse-reset --output artifacts/shd_canary.py --run-output shd_canary
```

The disjoint remaining range is `--start 100 --count 761`. The optional
`--spike-reader numpy-current` reads current-run spikes without rebuilding
recording history. `--repeat-first` records a separate repeated execution.
Preserve sample ranges, all primary outputs and actual runtime versions.

`collect_spinnaker1_campaign.py` collects existing submitted jobs without
resubmission. `summarize_spinnaker1_reports.py` extracts warning categories
without account paths. Neither empty per-input diagnostics nor a successful
job status alone establishes a complete, warning-free physical capture.

For a complete 861-input matrix, use `analyze_spinnaker1_matrix.py` with the
capture folders and `results/spinnaker1_shd` audit. Ground-truth labels are
optional and used only for post-capture accuracy. For the measurements supplied
here, use:

```sh
python experiments/summarize_spinnaker1_retained.py --verify-only
```

The distinct convolutional DVS mapping is documented in the
[DVS guide](SPINNAKER1_DVS_EXPERIMENT.md).
