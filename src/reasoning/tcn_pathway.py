from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime import LazyPathwayInference, RuntimeFactory, as_float32_array, validate_rank


class TcnPathwayInference(LazyPathwayInference):
    DEFAULT_MODEL_PATH = Path("models/onnx/tcn_pathway.onnx")

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            input_names=("sequence_features",),
            output_dim=64,
            runtime_factory=runtime_factory,
        )

    def _normalize_inputs(self, *inputs: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
        if len(inputs) != 1:
            raise ValueError("TCN pathway expects sequence_features")
        sequence_features = as_float32_array(inputs[0])
        validate_rank(sequence_features, 3, "sequence_features")
        if sequence_features.shape[1] != 128:
            raise ValueError("expected TCN input shape [B, 128, T]")
        return (sequence_features,), sequence_features.shape[0]
