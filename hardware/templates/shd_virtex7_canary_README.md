# Held-out SHD physical canary for Virtex-7

This bundle uses the same weights, numeric formats and neuron transitions as
`../shd_floor_q8q16_v1/`. No parameter file changes. It supplies new held-out
test inputs for the physical comparison in Table V, shared with the SpiNNaker-1
SHD campaign. The earlier training-pool batch remains separate RTL simulation
evidence.

## Board run

1. Keep the existing five-seed `unrepaired/` and `repaired/` parameter memory
   files from `shd_floor_q8q16_v1/seeds/`.
2. Replace the batch input memory with `common/canary_input_spikes.mem`.
   `common/canary_samples.csv` gives the sample order and first memory line.
3. Execute all 861 inputs for each seed and each repair state. Reset the network
   state before each input. Each sample still has 700 channels and 50 timesteps.
4. Preserve physical predictions and raw board traces, together with the board
   configuration, clock, firmware and bitstream filenames.
5. Compare against the corresponding `unrepaired_emulator_predictions` or
   `repaired_emulator_predictions` in `seeds/<seed>/golden_canary.npz`.

The compressed `common/canary_inputs.npz` contains the same binary events.
Channel zero remains the least significant bit of each 700-bit memory word.
The existing handoff documents parameter layout and exact integer arithmetic.

## Results and interpretation

Each seed's golden file contains sample IDs, original source predictions,
integer-emulator predictions, and final Q16 logits for both target models.
These are software outputs, not physical board measurements. Ground-truth
labels are absent.

For completed captures, aligned ground truth and five-seed source predictions
are available separately in the repository folder `hardware/evaluation/`.
That folder includes an accuracy and delta A evaluator.

The separate 861-input semantic audit is in `results/virtex7_shd/` at repository
root. Both splits come from the held-out SHD test pool and are disjoint. The
earlier RTL CSVs must not be joined to these different input IDs.

For each seed and repair state, return a compressed NumPy capture containing
`sample_ids`, `pair_ids`, and `predictions`. Use one primary board execution per
input, and keep additional repeated executions separate. Labels are used only
after the physical capture is complete to evaluate accuracy change.

Recreate this bundle and its semantic audit with
`python experiments/prepare_virtex7_shd_audit.py`. Physical data and measured
emulator-board disagreement are still needed to form the total certificate.

## Analyze completed board captures

Place each capture in `CAPTURES/<seed>/<variant>/capture.npz`, where `variant`
is `unrepaired` or `repaired`. Its adjacent `manifest.json` records the actual
board, firmware, bitstream, mapping and run information using
`schemas/HardwareRunManifest.schema.json`. Pair IDs can be ordinary run labels,
with one distinct label for each primary input execution.

```sh
python experiments/analyze_virtex7_shd.py --captures CAPTURES --audit results/virtex7_shd --output results/virtex7_shd/physical_certificate.json
```

The analysis aligns sample IDs, computes each condition's conformance term and
total bound, and reports five-seed means and budget verdicts. Add
`--labels data/processed/shd_v1/test.npz` for post-capture accuracy evaluation.
Missing conditions return progress counts without a physical certificate.
Retain the raw board traces alongside these compact captures.
