from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def config_description(value: Any) -> str:
    """Return a readable canonical description of a configuration."""

    return canonical_json(value)


def file_reference(path: str | Path) -> str:
    """Return the artifact filename used by a report or manifest."""

    return Path(path).name


def array_description(array: np.ndarray) -> str:
    """Describe an array by dtype and shape."""

    value = np.asarray(array)
    return f"{value.dtype}[{','.join(str(size) for size in value.shape)}]"


def code_revision(root: str | Path | None = None) -> str:
    """Return the commit id and disclose modified execution code.

    Generated reports and documentation are intentionally excluded from the
    dirty check. A manuscript rebuild must not change the revision attached to
    an otherwise identical experiment. Source, experiment runners, semantics
    schemas, hardware descriptions, and configuration files are included.
    Official evidence should contain a plain commit id. The ``+dirty`` suffix
    makes development evidence honest when any executable input is uncommitted.
    """

    execution_paths = (
        "src",
        "experiments",
        "configs",
        "schemas",
        "hardware",
        "rtl",
        "pyproject.toml",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *execution_paths,
            ],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return f"{revision}+dirty" if status.strip() else revision
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def write_json(path: str | Path, value: Any) -> Path:
    """Write a readable JSON report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
