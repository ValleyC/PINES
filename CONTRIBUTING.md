# Contributing

Changes to operational semantics require a schema-version decision, scalar and
vector transition tests, at least one handcrafted trace, and randomized
differential coverage. Changes to a frozen protocol create a new protocol file;
existing physical evidence is never relabeled under a revised protocol.

Before review, run:

```powershell
python -m pytest
transportcert reproduce --config configs/toy_study.json --output-dir artifacts/local-toy
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The last command runs from `paper/`. Evidence and reports are immutable: choose a
new output directory or run ID instead of overwriting an artifact.

