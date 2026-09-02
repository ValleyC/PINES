from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import file_reference, write_json
from ..semantics import ExecutionSemantics


@dataclass(frozen=True)
class NIRImportRecord:
    source_path: str
    source_file: str
    nir_version: str
    semantics_description: str
    assumptions: tuple[str, ...]


class NIRAdapter:
    """Thin adapter around the official NIR Python package.

    NIR stores graph structure and parameters; execution details that are not
    fixed by the graph live in an ExecutionSemantics sidecar.
    """

    @staticmethod
    def _module() -> Any:
        try:
            return importlib.import_module("nir")
        except ImportError as error:
            raise RuntimeError(
                "NIR support is optional; install pines-snn[nir] and pin the version"
            ) from error

    def load(
        self, path: str | Path, semantics: ExecutionSemantics
    ) -> tuple[Any, NIRImportRecord]:
        source = Path(path).resolve()
        module = self._module()
        version = importlib.metadata.version("nir")
        graph = module.read(str(source))
        record = NIRImportRecord(
            source_path=str(source),
            source_file=file_reference(source),
            nir_version=version,
            semantics_description=semantics.semantics_description,
            assumptions=(
                "graph parameters are interpreted under the attached semantics sidecar",
                "unsupported NIR primitives must be rejected by the backend adapter",
            ),
        )
        return graph, record

    def write_sidecar(
        self,
        graph_path: str | Path,
        semantics: ExecutionSemantics,
        output_path: str | Path,
    ) -> Path:
        graph_path = Path(graph_path).resolve()
        return write_json(
            output_path,
            {
                "schema_version": "NIRExecutionSidecar/v1",
                "graph_path": str(graph_path),
                "graph_file": file_reference(graph_path),
                "execution_semantics": semantics.to_dict(),
                "semantics_description": semantics.semantics_description,
            },
        )
