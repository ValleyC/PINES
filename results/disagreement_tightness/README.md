# Disagreement-bound sharpness

`witness_summary.json` and `witness_rows.csv` audit all 150 stored prediction
pairs from SHD, N-MNIST, and DVS Gesture: five seeds and ten semantic targets per
benchmark.

For each reference/target prediction pair, two synthetic label vectors are
constructed without reading dataset labels. Setting labels equal to reference
predictions makes reference accuracy exceed target accuracy by exactly the
disagreement rate. Setting labels equal to target predictions attains the same
gap in the opposite direction. Every cell attains both witnesses.

This verifies the elementary sharpness result on the actual experiment
artifacts: disagreement is a valid but information-theoretically unimprovable
distribution-free upper bound when only paired predictions are available. Any
tighter accuracy-change certificate must use additional assumptions or labels.

Reproduce with:

```powershell
python experiments/analyze_disagreement_tightness.py
```
