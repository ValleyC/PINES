from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

from ..semantics import ExecutionSemantics


@dataclass(frozen=True)
class SpiNNaker2Conversion:
    network: Any
    input_population: Any
    output_population: Any
    py_spinnaker2_version: str
    semantics_description: str
    conversion_config: dict[str, Any]


class SpiNNaker2Adapter:
    """Official NIR conversion path with explicit version and semantic capture."""

    def convert(
        self,
        nir_graph: Any,
        semantics: ExecutionSemantics,
        config: dict[str, Any] | None = None,
        required_version: str | None = None,
    ) -> SpiNNaker2Conversion:
        try:
            s2_nir = importlib.import_module("spinnaker2.s2_nir")
            version = importlib.metadata.version("py-spinnaker2")
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise RuntimeError(
                "SpiNNaker2 support requires a pinned py-spinnaker2 installation"
            ) from error
        if required_version is not None and version != required_version:
            raise RuntimeError(
                f"py-spinnaker2 version {version} does not match pin {required_version}"
            )
        conversion_config = dict(config or {})
        network, input_population, output_population = s2_nir.from_nir(
            nir_graph, config=conversion_config or None
        )
        return SpiNNaker2Conversion(
            network=network,
            input_population=input_population,
            output_population=output_population,
            py_spinnaker2_version=version,
            semantics_description=semantics.semantics_description,
            conversion_config=conversion_config,
        )

