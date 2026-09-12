"""Job packaging tests do not run a simulator or create physical evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


spec = importlib.util.spec_from_file_location("build_nmpi_job",
    Path(__file__).resolve().parents[1] / "experiments/build_spinnaker1_nmpi_job.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_embedded_job_preserves_models_and_passes_run_arguments(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "src/pines").mkdir(parents=True)
    (runtime / "src/pines/__init__.py").write_text("")
    (runtime / "experiments").mkdir()
    (runtime / "experiments/run_spinnaker1_matrix.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path('observed_arguments.json').write_text(json.dumps(sys.argv))\n")
    (runtime / "experiments/run_spinnaker1_shd.py").write_text("")
    bundle = tmp_path / "bundle"
    (bundle / "models/1701").mkdir(parents=True)
    for variant in builder.VARIANTS:
        (bundle / f"models/1701/{variant}.npz").write_bytes(variant.encode())
    inputs = bundle / "development_inputs.npz"
    inputs.write_bytes(b"frozen-input-bytes")
    (bundle / "DATA_LICENSE.md").write_text("Dataset attribution")
    source = builder.build_job(runtime, bundle, inputs, seeds=[1701],
                               start=2, count=3, time_scale_factor=100)
    job = tmp_path / "job.py"
    job.write_text(source)
    result = subprocess.run([sys.executable, str(job)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "observed_arguments.json").read_text())
    assert args[args.index("--start") + 1] == "2"
    assert args[args.index("--count") + 1] == "3"
    assert args[args.index("--time-scale-factor") + 1] == "100"
    assert args[args.index("--seeds") + 1:] == ["1701"]
    unpacked = tmp_path / "pines_job_payload/bundle"
    assert (unpacked / inputs.name).read_bytes() == inputs.read_bytes()
    for variant in builder.VARIANTS:
        name = f"models/1701/{variant}.npz"
        assert (unpacked / name).read_bytes() == (bundle / name).read_bytes()
    assert (unpacked / "DATA_LICENSE.md").read_text() == "Dataset attribution"
