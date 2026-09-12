import importlib.util
import json
from pathlib import Path
import sys
import zipfile


scripts = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location("campaign_collection", scripts / "collect_spinnaker1_campaign.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
finally:
    sys.path.pop(0)


def test_collector_waits_on_existing_handle_and_keeps_repeats_separate(tmp_path, monkeypatch):
    calls, analyses = [], []

    class Client:
        def get_job(self, job, with_log):
            calls.append((job, with_log))
            if len(calls) == 1:
                return dict(status="running", log="1/2 shd-test-00001: completed")
            return dict(status="finished", log="completed", output_data=dict(files=[
                dict(url="https://example.invalid/own_capture.zip"),
                dict(url="https://example.invalid/reports.zip")]))

    def download(url, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("input_00000/summary.json", "{}")
            archive.writestr("repeat_first/summary.json", "{}")

    def analyze(paths, audit):
        analyses.append(paths)
        assert (paths[0] / "input_00000/summary.json").exists()
        return dict(physical_certificate=True)

    monkeypatch.setattr(collector, "analyze_shd", analyze)
    monkeypatch.setattr(collector, "analyze_repeats", lambda paths, task: dict(physical_certificate=False))
    progress = collector.collect_campaign(Client(), [dict(job="existing-job", task="shd", name="batch")],
        tmp_path / "collected", dict(shd=tmp_path / "audit"), download=download, sleep=lambda _: None)
    assert calls == [("existing-job", True), ("existing-job", True)]
    assert len(analyses) == 1
    assert progress["status"] == "collection_finished"
    assert json.loads((tmp_path / "collected/shd_repeat_analysis.json").read_text())["physical_certificate"] is False


def test_terminal_job_without_capture_remains_missing_data(tmp_path, monkeypatch):
    class Client:
        def get_job(self, job, with_log):
            return dict(status="error", log="allocation failed", output_data=None)

    monkeypatch.setattr(collector, "analyze_dvs", lambda paths, audit:
        dict(status="incomplete_capture", physical_certificate=False, observations=len(paths)))
    monkeypatch.setattr(collector, "analyze_repeats", lambda *args: {})
    collector.collect_campaign(Client(), [dict(job="failed-job", task="dvs", name="batch")],
        tmp_path, dict(dvs=tmp_path / "audit"), sleep=lambda _: None)
    report = json.loads((tmp_path / "dvs_label_free_analysis.json").read_text())
    assert report["observations"] == 0
    assert report["physical_certificate"] is False


def test_log_failure_falls_back_to_same_job_without_false_empty_log(tmp_path, monkeypatch):
    calls = []

    class Client:
        def get_job(self, job, with_log):
            calls.append((job, with_log))
            if with_log:
                raise RuntimeError("Error 500: Internal Server Error")
            return dict(status="finished", output_data=None)

    monkeypatch.setattr(collector, "analyze_shd", lambda paths, audit: dict(status="incomplete_capture"))
    monkeypatch.setattr(collector, "analyze_repeats", lambda *args: {})
    progress = collector.collect_campaign(Client(), [dict(job="existing-job", task="shd", name="batch")],
        tmp_path, dict(shd=tmp_path / "audit"), sleep=lambda _: None)
    assert calls == [("existing-job", True), ("existing-job", False)]
    assert progress["jobs"]["batch"]["status"] == "finished"
    assert "log_retrieval_error" in progress["jobs"]["batch"]
    assert not (tmp_path / "batch/service.log").exists()
    assert (tmp_path / "batch/service_log_unavailable.txt").exists()
