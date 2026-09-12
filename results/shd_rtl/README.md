# SHD RTL simulation results

The ten CSV files contain 861 predictions for each of five training seeds,
before and after repair. All 8,610 class predictions match the frozen
floor-rounded fixed-point emulator outputs in the SHD hardware handoff bundle.
The supplied golden columns were also checked against that bundle, rather than
relying only on the CSV `match` column.

These are **RTL simulation results**, not physical Virtex-7 board captures.
They support prediction-level consistency of the simulated implementation on
these inputs. They do not supply a physical-device conformance term, establish
every internal state transition, or fill the manuscript's physical hardware row.

Recompute the comparison with:

```sh
python experiments/summarize_shd_rtl.py
```

`sample_index` is the row in the frozen bundle's audit input file. `rtl_prediction`
and `golden_prediction` are zero-based class indices. `match` is one when these
two columns agree. `summary.json` additionally compares both columns directly
against the bundle's emulator predictions.

The full-model batch testbench and simulator project accompanying these CSVs
have not yet been supplied. The summary is reproducible from the retained files,
but independently rerunning the reported full-network RTL simulation requires
those additional source files.
