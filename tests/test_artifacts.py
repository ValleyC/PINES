from __future__ import annotations

import subprocess

from pines.artifacts import code_revision


def _git(path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=path, text=True, stderr=subprocess.DEVNULL
    ).strip()


def test_code_revision_discloses_modified_execution_code(tmp_path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "PINES test")
    source = tmp_path / "src"
    source.mkdir()
    module = source / "executor.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "src/executor.py")
    _git(tmp_path, "commit", "-m", "initial")

    revision = _git(tmp_path, "rev-parse", "HEAD")
    assert code_revision(tmp_path) == revision

    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert code_revision(tmp_path) == f"{revision}+dirty"


def test_code_revision_ignores_generated_results(tmp_path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "PINES test")
    source = tmp_path / "src"
    source.mkdir()
    (source / "executor.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "src/executor.py")
    _git(tmp_path, "commit", "-m", "initial")

    revision = _git(tmp_path, "rev-parse", "HEAD")
    results = tmp_path / "results"
    results.mkdir()
    (results / "report.json").write_text("{}\n", encoding="utf-8")
    assert code_revision(tmp_path) == revision
