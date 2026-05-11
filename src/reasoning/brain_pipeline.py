from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from src.complexity.estimator import ComplexityEstimate
from src.perception.sensor_fusion import FusedEntity
from src.router.adaptive_router import AdaptiveRouter, ReasoningPathway, RoutingDecision

from .attention_pathway import AttentionPathwayInference
from .fusion import GatedFusionInference
from .gnn_pathway import GnnPathwayInference
from .gru_pathway import GruPathwayInference
from .tcn_pathway import TcnPathwayInference


class ComplexityEstimatorProtocol(Protocol):
    def estimate_from_entities(
        self,
        entities: Sequence[FusedEntity],
        *,
        soh_budget: float,
        anomaly_score_prev: float = 0.0,
        scene_embedding: np.ndarray | None = None,
        max_entities: int = 20,
    ) -> ComplexityEstimate:
        """Estimate scene complexity from perception entities."""


class PathwayInferenceProtocol(Protocol):
    def infer(self, *inputs: np.ndarray) -> np.ndarray:
        """Run one reasoning pathway."""


class FusionInferenceProtocol(Protocol):
    def infer(
        self,
        pathway_outputs: Mapping[ReasoningPathway | str, np.ndarray] | np.ndarray,
        active_pathways: Sequence[ReasoningPathway | str] | None = None,
    ) -> np.ndarray:
        """Fuse active pathway outputs."""


@dataclass(frozen=True, slots=True)
class BrainPipelineResult:
    estimate: ComplexityEstimate
    routing: RoutingDecision
    pathway_outputs: dict[ReasoningPathway, np.ndarray]
    unified_output: np.ndarray
    latency_ms: float

    @property
    def within_latency_budget(self) -> bool:
        return self.latency_ms <= self.routing.latency_budget_ms


class AdaptiveBrainPipeline:
    def __init__(
        self,
        estimator: ComplexityEstimatorProtocol,
        *,
        router: AdaptiveRouter | None = None,
        pathways: Mapping[ReasoningPathway, PathwayInferenceProtocol] | None = None,
        fusion: FusionInferenceProtocol | None = None,
    ) -> None:
        self.estimator = estimator
        self.router = router or AdaptiveRouter()
        self.pathways = dict(pathways) if pathways is not None else _default_pathways()
        self.fusion = fusion or GatedFusionInference()

    def run(
        self,
        entities: Sequence[FusedEntity],
        sequence_features: np.ndarray,
        entity_features: np.ndarray,
        adjacency: np.ndarray,
        *,
        soh_budget: float,
        anomaly_score_prev: float = 0.0,
        scene_embedding: np.ndarray | None = None,
    ) -> BrainPipelineResult:
        start = perf_counter()
        estimate = self.estimator.estimate_from_entities(
            entities,
            soh_budget=soh_budget,
            anomaly_score_prev=anomaly_score_prev,
            scene_embedding=scene_embedding,
        )
        routing = self.router.route(estimate.level, soh_budget=soh_budget)
        pathway_outputs = self._run_active_pathways(
            routing.active_pathways,
            sequence_features=sequence_features,
            entity_features=entity_features,
            adjacency=adjacency,
        )
        unified_output = self.fusion.infer(pathway_outputs)
        return BrainPipelineResult(
            estimate=estimate,
            routing=routing,
            pathway_outputs=pathway_outputs,
            unified_output=unified_output,
            latency_ms=(perf_counter() - start) * 1000.0,
        )

    def _run_active_pathways(
        self,
        active_pathways: Sequence[ReasoningPathway],
        *,
        sequence_features: np.ndarray,
        entity_features: np.ndarray,
        adjacency: np.ndarray,
    ) -> dict[ReasoningPathway, np.ndarray]:
        outputs: dict[ReasoningPathway, np.ndarray] = {}
        for pathway in active_pathways:
            runner = self.pathways[pathway]
            if pathway == ReasoningPathway.GRU:
                outputs[pathway] = runner.infer(sequence_features)
            elif pathway == ReasoningPathway.TCN:
                outputs[pathway] = runner.infer(np.swapaxes(sequence_features, 1, 2))
            elif pathway == ReasoningPathway.ATTENTION:
                outputs[pathway] = runner.infer(entity_features)
            elif pathway == ReasoningPathway.GNN:
                outputs[pathway] = runner.infer(entity_features, adjacency)
            else:
                raise ValueError(f"unsupported pathway: {pathway}")
        return outputs


def _default_pathways() -> dict[ReasoningPathway, PathwayInferenceProtocol]:
    return {
        ReasoningPathway.GRU: GruPathwayInference(),
        ReasoningPathway.TCN: TcnPathwayInference(),
        ReasoningPathway.ATTENTION: AttentionPathwayInference(),
        ReasoningPathway.GNN: GnnPathwayInference(),
    }
