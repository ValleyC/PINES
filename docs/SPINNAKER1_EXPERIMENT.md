# SpiNNaker-1 experiment

## Current campaign

The full SHD hardware campaign was launched through EBRAINS on 12 September
2026 UTC: 861 held-out inputs, five training seeds, and original/repaired models,
giving 8,610 input-condition observations. One fresh allocation serves all ten
conditions per input. Hidden recurrence executes on SpiNNaker-1. The original
linear readout is computed on the host from captured spikes, a hybrid backend.

The remote output directory is
`PINES_spinnaker1_runs/shd_canary_five_seeds_v1/`, with a sibling `.log`.
The frozen execution bundle is `spinnaker1_ebrains_bundle_v8`.
Completion must be established from the process and complete captures, not from
this document. At approximately 131 seconds per input, the sequential campaign
takes about 31 hours before allocation or service delays.

The submission manuscript is `transport_manuscript/`. DVS Gesture hardware
execution and physical Virtex-7 captures remain outstanding. The peer's matching
SHD predictions are RTL simulation evidence in `results/shd_rtl/`.
The DVS mapping and development progress are described in
[`SPINNAKER1_DVS_EXPERIMENT.md`](SPINNAKER1_DVS_EXPERIMENT.md).

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
