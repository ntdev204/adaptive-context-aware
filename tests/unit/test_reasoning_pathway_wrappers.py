from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.reasoning import (
    AttentionPathwayInference,
    GnnPathwayInference,
    GruPathwayInference,
    TcnPathwayInference,
)


class FakeRuntime:
    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim
        self.calls = 0

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.zeros((inputs[0].shape[0], self.output_dim), dtype=np.float32)


class CountingRuntimeFactory:
    OUTPUT_DIMS = {
        "gru_pathway.engine": 64,
        "tcn_pathway.engine": 64,
        "attention_pathway.engine": 128,
        "gnn_pathway.engine": 256,
    }

    def __init__(self) -> None:
        self.created: list[Path] = []
        self.runtimes: list[FakeRuntime] = []

    def __call__(self, model_path: Path, input_names: tuple[str, ...]) -> FakeRuntime:
        del input_names
        runtime = FakeRuntime(self.OUTPUT_DIMS[model_path.name])
        self.created.append(model_path)
        self.runtimes.append(runtime)
        return runtime


def test_reasoning_wrappers_are_lazy_loaded() -> None:
    factory = CountingRuntimeFactory()
    gru = GruPathwayInference(runtime_factory=factory)
    tcn = TcnPathwayInference(runtime_factory=factory)

    assert not gru.is_loaded
    assert not tcn.is_loaded

    output = gru.infer(np.zeros((2, 8, 128), dtype=np.float32))

    assert output.shape == (2, 64)
    assert gru.is_loaded
    assert not tcn.is_loaded
    assert [path.name for path in factory.created] == ["gru_pathway.engine"]

    gru.infer(np.zeros((2, 8, 128), dtype=np.float32))

    assert len(factory.created) == 1
    assert factory.runtimes[0].calls == 2


def test_reasoning_wrapper_output_contracts() -> None:
    factory = CountingRuntimeFactory()

    assert GruPathwayInference(runtime_factory=factory).infer(np.zeros((2, 8, 128))).shape == (2, 64)
    assert TcnPathwayInference(runtime_factory=factory).infer(np.zeros((2, 128, 8))).shape == (2, 64)
    assert AttentionPathwayInference(runtime_factory=factory).infer(np.zeros((2, 5, 128))).shape == (2, 128)
    assert GnnPathwayInference(runtime_factory=factory).infer(np.zeros((2, 5, 128)), np.ones((2, 5, 5))).shape == (
        2,
        256,
    )


def test_gnn_wrapper_accepts_unbatched_adjacency() -> None:
    output = GnnPathwayInference(runtime_factory=CountingRuntimeFactory()).infer(
        np.zeros((3, 4, 128), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
    )

    assert output.shape == (3, 256)


def test_reasoning_wrappers_reject_wrong_shapes() -> None:
    factory = CountingRuntimeFactory()

    with pytest.raises(ValueError):
        GruPathwayInference(runtime_factory=factory).infer(np.zeros((2, 8, 127)))
    with pytest.raises(ValueError):
        TcnPathwayInference(runtime_factory=factory).infer(np.zeros((2, 127, 8)))
    with pytest.raises(ValueError):
        AttentionPathwayInference(runtime_factory=factory).infer(np.zeros((2, 0, 128)))
    with pytest.raises(ValueError):
        GnnPathwayInference(runtime_factory=factory).infer(np.zeros((2, 5, 128)), np.ones((4, 4)))


def test_reasoning_wrapper_rejects_runtime_output_shape_mismatch() -> None:
    class BadRuntime:
        def run(self, *inputs: np.ndarray) -> np.ndarray:
            return np.zeros((inputs[0].shape[0], 65), dtype=np.float32)

    def bad_factory(model_path: Path, input_names: tuple[str, ...]) -> BadRuntime:
        del model_path, input_names
        return BadRuntime()

    with pytest.raises(ValueError):
        GruPathwayInference(runtime_factory=bad_factory).infer(np.zeros((2, 8, 128)))
