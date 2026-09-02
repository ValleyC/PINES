# Contributing

This repository is frozen as an anonymous review artifact. Changes that affect
reported evidence must create a new versioned result instead of overwriting an
existing artifact.

Before proposing a change, run:

```powershell
python -m pytest
python experiments/verify_manuscript_results.py
pines reproduce --config configs/toy_study.json --output-dir artifacts/local-toy
```

Changes to execution semantics require a schema-version decision, scalar and
vector transition tests, at least one handcrafted trace, and randomized
differential coverage. Hardware evidence additionally requires a complete
manifest and capture filenames.
