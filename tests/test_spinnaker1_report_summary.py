import importlib.util
from pathlib import Path
import sqlite3
import zipfile


spec = importlib.util.spec_from_file_location("report_summary",
    Path(__file__).resolve().parents[1] / "experiments/summarize_spinnaker1_reports.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_final_provenance_warnings_are_counted_as_log_entries(tmp_path):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE p_log_provenance(level INTEGER,message TEXT)")
    connection.executemany("INSERT INTO p_log_provenance VALUES (?,?)", [
        (20, "informational only"),
        (30, "The extra monitor reports 20 packets were dumped from outgoing links"),
        (30, "The extra monitor reports 20 packets were dumped from outgoing links"),
        (30, "The transmission buffer was blocked on 12 occasions"),
        (40, "an execution error")])
    connection.commit()
    path = tmp_path / "reports.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("reports/session/global_provenance.sqlite3", connection.serialize())
    connection.close()
    report = module.summarize([path])
    row = report["archives"][0]
    assert row["warning_log_entries"] == 3
    assert row["error_or_critical_log_entries"] == 1
    assert row["categories"] == dict(router_packet_dump=2, transmission_backpressure=1, other=1)
    assert str(tmp_path) not in str(report)


def test_missing_log_database_is_not_reported_as_warning_free(tmp_path):
    path = tmp_path / "reports.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("reports/finished", "done")
    assert module.summarize([path])["archives"] == [
        dict(archive_index=0, status="no_global_log_database")]
