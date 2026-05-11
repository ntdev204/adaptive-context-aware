from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import LazyPathwayInference, RuntimeFactory, as_float32_array, validate_rank


class GnnPathwayInference(LazyPathwayInference):
    DEFAULT_MODEL_PATH = Path("models/engines/gnn_pathway.engine")

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            input_names=("entity_features", "adjacency"),
            output_dim=256,
            runtime_factory=runtime_factory,
        )

    def _normalize_inputs(self, *inputs: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
        if len(inputs) != 2:
            raise ValueError("GNN pathway expects entity_features and adjacency")
        entity_features = as_float32_array(inputs[0])
        adjacency = as_float32_array(inputs[1])
        validate_rank(entity_features, 3, "entity_features")
        if entity_features.shape[-1] != 128:
            raise ValueError("expected GNN entity input shape [B, N, 128]")

        batch_size, entity_count, _ = entity_features.shape
        if adjacency.ndim == 2:
            if adjacency.shape != (entity_count, entity_count):
                raise ValueError("expected unbatched adjacency shape [N, N]")
            adjacency = np.broadcast_to(adjacency, (batch_size, entity_count, entity_count)).copy()
        elif adjacency.ndim == 3:
            if adjacency.shape != (batch_size, entity_count, entity_count):
                raise ValueError("expected batched adjacency shape [B, N, N]")
        else:
            raise ValueError("adjacency must have rank 2 or 3")
        return (entity_features, adjacency), batch_size
