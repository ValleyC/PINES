# Retained SpiNNaker primary measurements

These files support the SpiNNaker rows of manuscript Table V:

- SHD primary batch: first 100 canary inputs, five seeds, original/repaired models,
  1,000 recording-condition predictions.
- DVS primary batch: first ten four-window recordings, seed 1701, original/repaired
  models, 20 recording-condition predictions.

Every retained primary observation is included. The older SHD 75-input capture,
development probes, and separate repeats are not mixed into these results.
Labels are used only to evaluate the fixed captured observations, not for repair,
sample selection, or certificate construction.

## Results

On the retained SHD inputs, mean source accuracy is 73.4%; hardware accuracy is
62.8% before repair and 72.4% afterward. Mean absolute per-seed accuracy change
is 10.6/2.2 points. The repaired 2.2 is a mean of absolute per-seed changes,
not the 1.0-point difference between mean accuracies. Source-to-device
disagreement is 29.2/14.6%, and emulator-device disagreement is 14.2/10.4%.

DVS seed 1701 has source accuracy 90%, hardware accuracy 50/90%, absolute
accuracy change 40/0 points, and emulator-device disagreement 10/0%. Its
semantic bounds are the matching seed's 75.3/27.2 points, not the five-seed
84.2/33.3-point means used for the full planned matrix.

## Population bounds

The returned batches are prefixes of dataset-index-sorted canary sets, not
independently sampled smaller canaries. Keep the original planned sample counts
N=861 for SHD and N=160 for DVS. With n observed pairs and k measured emulator
disagreements, the completed canary's failure count is at most k+(N-n).
Monotonicity of the upper Clopper-Pearson limit gives

`U_hw = CP_upper(k + N - n, N, 0.05 / 80)`.

This covers every assignment to the unresolved pairs under the original
sampling and independent-execution assumptions. It does not assume the returned
prefix is representative or estimate missing measurements. The original
40-cell/two-term confidence correction is unchanged. Total bounds are capped
at one and return rejection at all three budgets for these SpiNNaker rows.

## Files and reproduction

- `observations.csv`: sample IDs, configuration, frozen source/emulator and
  captured device predictions, and post-capture evaluation labels.
- `shd_captured_traces.npz`: aligned hidden-spike traces and recorded logits
  for every listed SHD observation.
- `dvs_captured_traces.npz`: the four hidden-spike traces and window logits per
  DVS observation. Window aggregation is checked against the recorded decisions.
- `summary.json`: per-condition measurements, original planned counts, unresolved
  counts, simultaneous bounds and execution profiles. Conditions without outputs
  have no empirical accuracy or disagreement estimate.

To regenerate from the retained raw job captures in the original workspace:

```sh
python experiments/summarize_spinnaker1_retained.py
```

Check the published CSV/traces, predictions, and summary arithmetic without
the original job folders or datasets:

```sh
python experiments/summarize_spinnaker1_retained.py --verify-only
```

Execution logs include
dense-source warnings for SHD and router-dump/backpressure warnings for DVS.
All primary observations are retained regardless of these warnings. No
run-to-run variance estimate is inferred from these single-run primary batches.
