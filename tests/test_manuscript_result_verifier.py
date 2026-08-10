from __future__ import annotations

from experiments.verify_manuscript_results import file_hash_candidates


def test_csv_hash_candidates_are_line_ending_portable(tmp_path, monkeypatch) -> None:
    from experiments import verify_manuscript_results

    monkeypatch.setattr(verify_manuscript_results, "ROOT", tmp_path)
    csv_path = tmp_path / "rows.csv"
    csv_path.write_bytes(b"a,b\n1,2\n")
    lf_candidates = file_hash_candidates("rows.csv")
    csv_path.write_bytes(b"a,b\r\n1,2\r\n")
    crlf_candidates = file_hash_candidates("rows.csv")
    assert lf_candidates == crlf_candidates


def test_csv_hash_candidates_still_detect_content_changes(tmp_path, monkeypatch) -> None:
    from experiments import verify_manuscript_results

    monkeypatch.setattr(verify_manuscript_results, "ROOT", tmp_path)
    csv_path = tmp_path / "rows.csv"
    csv_path.write_bytes(b"a,b\n1,2\n")
    original = file_hash_candidates("rows.csv")
    csv_path.write_bytes(b"a,b\n1,3\n")
    changed = file_hash_candidates("rows.csv")
    assert original.isdisjoint(changed)
