"""Job packaging tests do not run a simulator or create physical evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


spec = importlib.util.spec_from_file_location("build_nmpi_job",
    Path(__file__).resolve().parents[1] / "experiments/build_spinnaker1_nmpi_job.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize("task", ["shd", "dvs", "dvs-reset"])
def test_embedded_job_preserves_models_and_passes_run_arguments(tmp_path, task):
    runtime = tmp_path / "runtime"
    (runtime / "src/pines").mkdir(parents=True)
    (runtime / "src/pines/__init__.py").write_text("")
    (runtime / "experiments").mkdir()
    fake_runner = (
        "import json, sys\nfrom pathlib import Path\n"
        "Path('observed_arguments.json').write_text(json.dumps(sys.argv))\n"
        "out=Path(sys.argv[sys.argv.index('--output')+1])\n"
        "for index in range(2):\n"
        "    folder=out/f'input_{index:05d}'\n"
        "    folder.mkdir(parents=True)\n"
        "    (folder/'summary.json').write_text(str(index))\n")
    (runtime / "experiments/run_spinnaker1_matrix.py").write_text(fake_runner)
    (runtime / "experiments/run_spinnaker1_dvs.py").write_text(fake_runner)
    (runtime / "experiments/probe_spinnaker1_dvs_reset.py").write_text(fake_runner)
    (runtime / "experiments/run_spinnaker1_shd.py").write_text("")
    bundle = tmp_path / "bundle"
    (bundle / "models/1701").mkdir(parents=True)
    variants = builder.VARIANTS if task == "shd" else ("original", "floor_repaired")
    for variant in variants:
        (bundle / f"models/1701/{variant}.npz").write_bytes(variant.encode())
        if task == "dvs-reset":
            traces = bundle / f"development_traces/1701/{variant}"
            traces.mkdir(parents=True)
            for index in range(2):
                (traces / f"input_{index:05d}.npz").write_bytes(b"calibration-trace")
    inputs = bundle / "development_inputs.npz"
    inputs.write_bytes(b"frozen-input-bytes")
    (bundle / "DATA_LICENSE.md").write_text("Dataset attribution")
    source = builder.build_job(runtime, bundle, inputs, seeds=[1701],
                               start=2, count=3, time_scale_factor=100,
                               task=task, windows=[0, 1, 2, 3], record_layers=True,
                               align_reset=task == "dvs-reset", reuse_reset=task != "dvs-reset",
                               repeat_first=task != "dvs-reset", spike_reader="numpy-current")
    job = tmp_path / "job.py"
    job.write_text(source)
    result = subprocess.run([sys.executable, str(job)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "observed_arguments.json").read_text())
    if task != "dvs-reset":
        assert args[args.index("--start") + 1] == "2"
        assert args[args.index("--count") + 1] == "3"
    assert args[args.index("--time-scale-factor") + 1] == "100"
    assert args[args.index("--spike-reader") + 1] == "numpy-current"
    if task == "shd":
        assert args[args.index("--seeds") + 1:] == ["1701", "--reuse-reset", "--repeat-first"]
    elif task == "dvs":
        assert args[args.index("--seed") + 1] == "1701"
        assert args[args.index("--windows") + 1:args.index("--windows") + 5] == ["0", "1", "2", "3"]
        assert "--record-layers" in args
        assert "--reuse-reset" in args
        assert "--repeat-first" in args
    else:
        assert args[args.index("--seed") + 1] == "1701"
        assert "--align-reset" in args
        assert "--inputs" not in args
        assert "--windows" not in args
    unpacked = tmp_path / "pines_job_payload/bundle"
    assert (unpacked / inputs.name).read_bytes() == inputs.read_bytes()
    for variant in variants:
        name = f"models/1701/{variant}.npz"
        assert (unpacked / name).read_bytes() == (bundle / name).read_bytes()
        if task == "dvs-reset":
            for index in range(2):
                name = f"development_traces/1701/{variant}/input_{index:05d}.npz"
                assert (unpacked / name).read_bytes() == b"calibration-trace"
    assert (unpacked / "DATA_LICENSE.md").read_text() == "Dataset attribution"
    with zipfile.ZipFile(tmp_path / "batch_shd_capture_capture.zip") as archive:
        assert archive.read("input_00000/summary.json") == b"0"
        assert archive.read("input_00001/summary.json") == b"1"
