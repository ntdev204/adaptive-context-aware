from __future__ import annotations

import numpy as np
import pytest

from src.complexity.estimator import ComplexityEstimate, ComplexityLevel
from src.reasoning.brain_pipeline import AdaptiveBrainPipeline
from src.router.adaptive_router import ReasoningPathway


class FixedEstimator:
    def __init__(self, level: ComplexityLevel) -> None:
        self.level = level

    def estimate_from_entities(self, *args: object, **kwargs: object) -> ComplexityEstimate:
        del args, kwargs
        probabilities = np.zeros(4, dtype=np.float32)
        probabilities[int(self.level)] = 1.0
        return ComplexityEstimate(
            level=self.level,
            logits=probabilities.copy(),
            probabilities=probabilities,
            input_vector=np.zeros(36, dtype=np.float32),
        )


class FakePathway:
    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim
        self.calls = 0

    def infer(self, *inputs: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.zeros((inputs[0].shape[0], self.output_dim), dtype=np.float32)


class FakeFusion:
    def __init__(self) -> None:
        self.last_pathways: tuple[ReasoningPathway, ...] = ()

    def infer(self, pathway_outputs: dict[ReasoningPathway, np.ndarray]) -> np.ndarray:
        self.last_pathways = tuple(pathway_outputs)
        batch_size = next(iter(pathway_outputs.values())).shape[0]
        return np.zeros((batch_size, 256), dtype=np.float32)


def _pipeline(level: ComplexityLevel) -> tuple[AdaptiveBrainPipeline, dict[ReasoningPathway, FakePathway], FakeFusion]:
    pathways = {
        ReasoningPathway.GRU: FakePathway(64),
        ReasoningPathway.TCN: FakePathway(64),
        ReasoningPathway.ATTENTION: FakePathway(128),
        ReasoningPathway.GNN: FakePathway(256),
    }
    fusion = FakeFusion()
    return AdaptiveBrainPipeline(FixedEstimator(level), pathways=pathways, fusion=fusion), pathways, fusion


@pytest.mark.parametrize(
    ("level", "expected_pathways"),
    [
        (ComplexityLevel.LOW, (ReasoningPathway.GRU,)),
        (ComplexityLevel.MED, (ReasoningPathway.GRU, ReasoningPathway.TCN)),
        (
            ComplexityLevel.HIGH,
            (ReasoningPathway.GRU, ReasoningPathway.TCN, ReasoningPathway.ATTENTION),
        ),
        (
            ComplexityLevel.CRITICAL,
            (
                ReasoningPathway.GRU,
                ReasoningPathway.TCN,
                ReasoningPathway.ATTENTION,
                ReasoningPathway.GNN,
            ),
        ),
    ],
)
def test_full_brain_pipeline_routes_each_complexity_level(
    level: ComplexityLevel,
    expected_pathways: tuple[ReasoningPathway, ...],
) -> None:
    pipeline, pathways, fusion = _pipeline(level)

    result = pipeline.run(
        [],
        sequence_features=np.zeros((2, 8, 128), dtype=np.float32),
        entity_features=np.zeros((2, 5, 128), dtype=np.float32),
        adjacency=np.ones((2, 5, 5), dtype=np.float32),
        soh_budget=1.0,
    )

    assert result.routing.active_pathways == expected_pathways
    assert tuple(result.pathway_outputs) == expected_pathways
    assert fusion.last_pathways == expected_pathways
    assert result.unified_output.shape == (2, 256)
    assert result.within_latency_budget
    for pathway, runner in pathways.items():
        assert runner.calls == (1 if pathway in expected_pathways else 0)


def test_full_brain_pipeline_applies_soh_downgrade() -> None:
    pipeline, _, _ = _pipeline(ComplexityLevel.CRITICAL)

    result = pipeline.run(
        [],
        sequence_features=np.zeros((1, 8, 128), dtype=np.float32),
        entity_features=np.zeros((1, 5, 128), dtype=np.float32),
        adjacency=np.ones((1, 5, 5), dtype=np.float32),
        soh_budget=0.4,
    )

    assert result.routing.requested_level == ComplexityLevel.CRITICAL
    assert result.routing.effective_level == ComplexityLevel.HIGH
    assert ReasoningPathway.GNN not in result.pathway_outputs
    assert result.unified_output.shape == (1, 256)
