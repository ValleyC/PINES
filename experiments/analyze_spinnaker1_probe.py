"""Check physical probe traces against the documented ideal IF_curr_delta update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_probe(directory: str | Path) -> dict:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    alpha = np.exp(-config["timestep_ms"] / config["tau_m_ms"])
    cases = []
    with np.load(directory / "traces.npz", allow_pickle=False) as raw:
        for spec in config["cases"]:
            prefix = spec["name"] + "__"
            v = raw[prefix + "v"].reshape(-1)
            exc = raw[prefix + "gsyn_exc"].reshape(-1)
            inh = raw[prefix + "gsyn_inh"].reshape(-1)
            times = raw[prefix + "times_ms"].reshape(-1)
            current = (exc - inh) / config["timestep_ms"] + spec["bias"]
            before_reset = current - alpha * (current - v)
            spikes = before_reset >= spec["threshold"]
            expected = np.where(spikes, spec["reset"], before_reset)
            # The last post-update voltage is not in the pre-update recording.
            residual = np.abs(v[1:] - expected[:-1])
            active = np.flatnonzero((exc != 0) | (inh != 0))
            first_delivery = float(times[active[0]]) if len(active) else None
            actual_spikes = raw[prefix + "spikes_ms"]
            expected_spikes = times[spikes]
            cases.append(dict(
                case=spec["name"], max_update_residual=float(residual.max()),
                spike_times_match=bool(np.array_equal(actual_spikes, expected_spikes)),
                first_delivery_ms=first_delivery,
                delivery_latency_steps=None if first_delivery is None else
                    (first_delivery - config["input_times_ms"][0]) / config["timestep_ms"]))
    impulse = next(c for c in cases if c["case"] == "impulse")
    delayed = next(c for c in cases if c["case"] == "delay_two")
    latency = impulse["delivery_latency_steps"]
    timing_ok = (latency is not None and float(latency).is_integer() and latency >= 1
                 and delayed["delivery_latency_steps"] == latency + 1)
    result = dict(
        stage="handcrafted_physical_probe", backend="spinnaker1",
        versions=summary["versions"], cases=cases,
        input_latency_steps=int(latency) if timing_ok else None,
        observed_timing_consistent=timing_ok,
        ideal_update_tolerance=0.001,
        ready_for_shd_smoke=bool(timing_ok and latency == 1 and all(
            c["max_update_residual"] <= 0.001 and c["spike_times_match"] for c in cases)),
        interpretation="Observed toy-trace conformance only. This is not an SHD certificate.")
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(analyze_probe(parser.parse_args().directory), indent=2))
