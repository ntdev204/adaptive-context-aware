from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.router.adaptive_router import ReasoningPathway
from src.utils.constants import UNIFIED_REASONING_DIM

from .runtime import PathwayRuntime, RuntimeFactory, TensorRTEngineRuntime, as_float32_array


class GatedFusionInference:
    DEFAULT_MODEL_PATH = Path("models/engines/gated_fusion.engine")
    PATHWAY_DIMS = {
        ReasoningPathway.GRU: 64,
        ReasoningPathway.TCN: 64,
        ReasoningPathway.ATTENTION: 128,
        ReasoningPathway.GNN: 256,
    }
    PATHWAY_ORDER = tuple(PATHWAY_DIMS)
    OUTPUT_DIM = UNIFIED_REASONING_DIM

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.runtime_factory = runtime_factory or TensorRTEngineRuntime
        self.input_names = ("pathway_outputs", "active_mask")
        self._offsets = self._build_offsets()
        self._runtime: PathwayRuntime | None = None

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def infer(
        self,
        pathway_outputs: Mapping[ReasoningPathway | str, np.ndarray] | np.ndarray,
        active_pathways: Sequence[ReasoningPathway | str] | None = None,
    ) -> np.ndarray:
        padded_outputs, active_mask, batch_size = self._prepare_inputs(pathway_outputs, active_pathways)
        output = np.asarray(self._get_runtime().run(padded_outputs, active_mask), dtype=np.float32)
        if output.shape != (batch_size, self.OUTPUT_DIM):
            raise ValueError("fusion runtime must return shape [B, 256]")
        return output

    def _prepare_inputs(
        self,
        pathway_outputs: Mapping[ReasoningPathway | str, np.ndarray] | np.ndarray,
        active_pathways: Sequence[ReasoningPathway | str] | None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if isinstance(pathway_outputs, Mapping):
            if active_pathways is not None:
                raise ValueError("active_pathways is only valid with concatenated tensor input")
            return self._prepare_mapping_input(pathway_outputs)
        if active_pathways is None:
            raise ValueError("active_pathways is required for concatenated tensor input")
        return self._prepare_concatenated_input(pathway_outputs, active_pathways)

    def _prepare_mapping_input(
        self,
        pathway_outputs: Mapping[ReasoningPathway | str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if not pathway_outputs:
            raise ValueError("fusion requires at least one active pathway")

        normalized_outputs = {
            self._coerce_pathway(name): as_float32_array(values) for name, values in pathway_outputs.items()
        }
        sample = next(iter(normalized_outputs.values()))
        batch_size = self._validate_output(sample, expected_dim=sample.shape[-1])
        padded_outputs = np.zeros((batch_size, sum(self.PATHWAY_DIMS.values())), dtype=np.float32)
        active_mask = np.zeros((batch_size, len(self.PATHWAY_ORDER)), dtype=np.float32)

        for index, pathway in enumerate(self.PATHWAY_ORDER):
            if pathway not in normalized_outputs:
                continue
            values = normalized_outputs[pathway]
            self._validate_output(values, expected_dim=self.PATHWAY_DIMS[pathway], batch_size=batch_size)
            start, end = self._offsets[pathway]
            padded_outputs[:, start:end] = values
            active_mask[:, index] = 1.0

        return padded_outputs, active_mask, batch_size

    def _prepare_concatenated_input(
        self,
        pathway_outputs: np.ndarray,
        active_pathways: Sequence[ReasoningPathway | str],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if isinstance(active_pathways, str):
            raise ValueError("active_pathways must be a sequence of pathway names")
        active_names = [self._coerce_pathway(name) for name in active_pathways]
        if not active_names:
            raise ValueError("fusion requires at least one active pathway")
        if len(set(active_names)) != len(active_names):
            raise ValueError("active_pathways must not contain duplicates")

        expected_dim = sum(self.PATHWAY_DIMS[name] for name in active_names)
        concat = as_float32_array(pathway_outputs)
        batch_size = self._validate_output(concat, expected_dim=expected_dim)
        padded_outputs = np.zeros((batch_size, sum(self.PATHWAY_DIMS.values())), dtype=np.float32)
        active_mask = np.zeros((batch_size, len(self.PATHWAY_ORDER)), dtype=np.float32)

        input_offset = 0
        for pathway in active_names:
            dim = self.PATHWAY_DIMS[pathway]
            start, end = self._offsets[pathway]
            padded_outputs[:, start:end] = concat[:, input_offset : input_offset + dim]
            active_mask[:, self.PATHWAY_ORDER.index(pathway)] = 1.0
            input_offset += dim

        return padded_outputs, active_mask, batch_size

    def _get_runtime(self) -> PathwayRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory(self.model_path, self.input_names)
        return self._runtime

    @staticmethod
    def _validate_output(values: np.ndarray, expected_dim: int, batch_size: int | None = None) -> int:
        if values.ndim != 2 or values.shape[-1] != expected_dim:
            raise ValueError(f"expected pathway output shape [B, {expected_dim}]")
        if batch_size is not None and values.shape[0] != batch_size:
            raise ValueError("all pathway outputs must share the same batch size")
        return values.shape[0]

    @classmethod
    def _coerce_pathway(cls, pathway: ReasoningPathway | str) -> ReasoningPathway:
        try:
            return ReasoningPathway(pathway)
        except ValueError as exc:
            raise ValueError(f"unknown pathway: {pathway}") from exc

    @classmethod
    def _build_offsets(cls) -> dict[ReasoningPathway, tuple[int, int]]:
        offsets: dict[ReasoningPathway, tuple[int, int]] = {}
        cursor = 0
        for pathway, dim in cls.PATHWAY_DIMS.items():
            offsets[pathway] = (cursor, cursor + dim)
            cursor += dim
        return offsets
