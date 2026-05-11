from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import LazyPathwayInference, RuntimeFactory, as_float32_array, validate_rank


class AttentionPathwayInference(LazyPathwayInference):
    DEFAULT_MODEL_PATH = Path("models/engines/attention_pathway.engine")

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            input_names=("entity_features",),
            output_dim=128,
            runtime_factory=runtime_factory,
        )

    def _normalize_inputs(self, *inputs: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
        if len(inputs) != 1:
            raise ValueError("attention pathway expects entity_features")
        entity_features = as_float32_array(inputs[0])
        validate_rank(entity_features, 3, "entity_features")
        if entity_features.shape[-1] != 128:
            raise ValueError("expected attention input shape [B, N, 128]")
        if entity_features.shape[1] <= 0:
            raise ValueError("attention pathway requires at least one entity")
        return (entity_features,), entity_features.shape[0]
