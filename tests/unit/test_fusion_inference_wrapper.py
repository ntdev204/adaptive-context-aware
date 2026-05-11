from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.reasoning.fusion import GatedFusionInference
from src.router.adaptive_router import ReasoningPathway


class FakeFusionRuntime:
    def __init__(self) -> None:
        self.calls = 0
        self.last_inputs: tuple[np.ndarray, np.ndarray] | None = None

    def run(self, *inputs: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.last_inputs = (inputs[0].copy(), inputs[1].copy())
        return np.zeros((inputs[0].shape[0], 256), dtype=np.float32)


class CountingFusionFactory:
    def __init__(self) -> None:
        self.created: list[Path] = []
        self.runtime = FakeFusionRuntime()

    def __call__(self, model_path: Path, input_names: tuple[str, ...]) -> FakeFusionRuntime:
        assert input_names == ("pathway_outputs", "active_mask")
        self.created.append(model_path)
        return self.runtime


def test_fusion_wrapper_is_lazy_loaded() -> None:
    factory = CountingFusionFactory()
    fusion = GatedFusionInference(runtime_factory=factory)

    assert not fusion.is_loaded

    output = fusion.infer({"gru": np.zeros((2, 64), dtype=np.float32)})

    assert output.shape == (2, 256)
    assert fusion.is_loaded
    assert [path.name for path in factory.created] == ["gated_fusion.engine"]
    assert factory.runtime.calls == 1


def test_fusion_wrapper_keeps_output_dim_for_variable_active_pathways() -> None:
    fusion = GatedFusionInference(runtime_factory=CountingFusionFactory())

    assert fusion.infer({"gru": np.zeros((2, 64), dtype=np.float32)}).shape == (2, 256)
    assert fusion.infer(
        {
            ReasoningPathway.GRU: np.zeros((2, 64), dtype=np.float32),
            ReasoningPathway.TCN: np.zeros((2, 64), dtype=np.float32),
            ReasoningPathway.ATTENTION: np.zeros((2, 128), dtype=np.float32),
        }
    ).shape == (2, 256)
    assert fusion.infer(
        {
            ReasoningPathway.GRU: np.zeros((2, 64), dtype=np.float32),
            ReasoningPathway.TCN: np.zeros((2, 64), dtype=np.float32),
            ReasoningPathway.ATTENTION: np.zeros((2, 128), dtype=np.float32),
            ReasoningPathway.GNN: np.zeros((2, 256), dtype=np.float32),
        }
    ).shape == (2, 256)


def test_fusion_wrapper_builds_active_mask() -> None:
    factory = CountingFusionFactory()
    fusion = GatedFusionInference(runtime_factory=factory)

    fusion.infer(
        np.zeros((3, 64 + 256), dtype=np.float32),
        active_pathways=(ReasoningPathway.GRU, ReasoningPathway.GNN),
    )

    assert factory.runtime.last_inputs is not None
    _, active_mask = factory.runtime.last_inputs
    assert active_mask.shape == (3, 4)
    assert active_mask[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert active_mask[:, 3].tolist() == [1.0, 1.0, 1.0]
    assert active_mask[:, 1:3].sum() == 0.0


def test_fusion_wrapper_rejects_invalid_inputs() -> None:
    fusion = GatedFusionInference(runtime_factory=CountingFusionFactory())

    with pytest.raises(ValueError):
        fusion.infer({})
    with pytest.raises(ValueError):
        fusion.infer({"gru": np.zeros((2, 63), dtype=np.float32)})
    with pytest.raises(ValueError):
        fusion.infer(
            {
                "gru": np.zeros((2, 64), dtype=np.float32),
                "tcn": np.zeros((3, 64), dtype=np.float32),
            }
        )
    with pytest.raises(ValueError):
        fusion.infer(np.zeros((2, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        fusion.infer(np.zeros((2, 64), dtype=np.float32), active_pathways=("gru", "gru"))
    with pytest.raises(ValueError):
        fusion.infer({"unknown": np.zeros((2, 16), dtype=np.float32)})


def test_fusion_wrapper_rejects_runtime_output_shape_mismatch() -> None:
    class BadRuntime:
        def run(self, *inputs: np.ndarray) -> np.ndarray:
            return np.zeros((inputs[0].shape[0], 128), dtype=np.float32)

    def bad_factory(model_path: Path, input_names: tuple[str, ...]) -> BadRuntime:
        del model_path, input_names
        return BadRuntime()

    with pytest.raises(ValueError):
        GatedFusionInference(runtime_factory=bad_factory).infer({"gru": np.zeros((2, 64), dtype=np.float32)})
