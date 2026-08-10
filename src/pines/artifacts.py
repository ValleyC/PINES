from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = canonical_json(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)}
    ).encode("utf-8")
    return sha256_bytes(header + contiguous.tobytes())


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


def write_json_immutable(path: str | Path, value: Any) -> Path:
    """Create a JSON artifact exactly once.

    The final creation uses an exclusive filesystem operation, so concurrent or
    accidental reruns cannot silently replace evidence.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, destination)
    except FileExistsError:
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
