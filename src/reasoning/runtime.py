from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np


class PathwayRuntime(Protocol):
    def run(self, *inputs: np.ndarray) -> np.ndarray:
        """Return one pathway output batch."""


RuntimeFactory = Callable[[Path, tuple[str, ...]], PathwayRuntime]


class OnnxPathwayRuntime:
    def __init__(self, model_path: Path, input_names: tuple[str, ...]) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"pathway ONNX model not found: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to load pathway ONNX models") from exc

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        session_inputs = self.session.get_inputs()
        self.input_names = input_names or tuple(input_.name for input_ in session_inputs)

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != len(self.input_names):
            raise ValueError(f"expected {len(self.input_names)} inputs, got {len(inputs)}")
        feed = {
            name: values.astype(np.float32, copy=False)
            for name, values in zip(self.input_names, inputs, strict=True)
        }
        return self.session.run(None, feed)[0]


class LazyPathwayInference:
    def __init__(
        self,
        model_path: str | Path,
        input_names: tuple[str, ...],
        output_dim: int,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_names = input_names
        self.output_dim = output_dim
        self.runtime_factory = runtime_factory or OnnxPathwayRuntime
        self._runtime: PathwayRuntime | None = None

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def infer(self, *inputs: np.ndarray) -> np.ndarray:
        normalized_inputs, batch_size = self._normalize_inputs(*inputs)
        output = np.asarray(self._get_runtime().run(*normalized_inputs), dtype=np.float32)
        if output.shape != (batch_size, self.output_dim):
            raise ValueError(f"pathway runtime must return shape [B, {self.output_dim}]")
        return output

    def _normalize_inputs(self, *inputs: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
        raise NotImplementedError

    def _get_runtime(self) -> PathwayRuntime:
        if self._runtime is None:
            self._runtime = self.runtime_factory(self.model_path, self.input_names)
        return self._runtime


def as_float32_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def validate_rank(values: np.ndarray, expected_rank: int, name: str) -> None:
    if values.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}")
