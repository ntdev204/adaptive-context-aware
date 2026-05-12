from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.complexity.estimator import ComplexityLevel


class ReasoningPathway(StrEnum):
    GRU = "gru"
    TCN = "tcn"
    ATTENTION = "attention"
    GNN = "gnn"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    requested_level: ComplexityLevel
    effective_level: ComplexityLevel
    active_pathways: tuple[ReasoningPathway, ...]
    latency_budget_ms: float
    pathway_budget_ms: dict[ReasoningPathway, float]
    soh_budget: float

    @property
    def downgraded(self) -> bool:
        return self.effective_level < self.requested_level


class AdaptiveRouter:
    SOH_DOWNGRADE_THRESHOLD = 0.5
    ROUTING_TABLE = {
        ComplexityLevel.LOW: (ReasoningPathway.GRU,),
        ComplexityLevel.MED: (ReasoningPathway.GRU, ReasoningPathway.TCN),
        ComplexityLevel.HIGH: (ReasoningPathway.GRU, ReasoningPathway.TCN, ReasoningPathway.ATTENTION),
        ComplexityLevel.CRITICAL: (
            ReasoningPathway.GRU,
            ReasoningPathway.TCN,
            ReasoningPathway.ATTENTION,
            ReasoningPathway.GNN,
        ),
    }
    LATENCY_BUDGET_MS = {
        ComplexityLevel.LOW: 5.0,
        ComplexityLevel.MED: 10.0,
        ComplexityLevel.HIGH: 20.0,
        ComplexityLevel.CRITICAL: 35.0,
    }
    PATHWAY_TARGET_MS = {
        ReasoningPathway.GRU: 2.0,
        ReasoningPathway.TCN: 3.0,
        ReasoningPathway.ATTENTION: 8.0,
        ReasoningPathway.GNN: 15.0,
    }

    def route(self, complexity_level: ComplexityLevel | int, soh_budget: float) -> RoutingDecision:
        requested_level = self._coerce_level(complexity_level)
        clipped_soh = float(np.clip(soh_budget, 0.0, 1.0))
        effective_level = self._apply_soh_override(requested_level, clipped_soh)
        active_pathways = self.ROUTING_TABLE[effective_level]
        return RoutingDecision(
            requested_level=requested_level,
            effective_level=effective_level,
            active_pathways=active_pathways,
            latency_budget_ms=self.LATENCY_BUDGET_MS[effective_level],
            pathway_budget_ms={pathway: self.PATHWAY_TARGET_MS[pathway] for pathway in active_pathways},
            soh_budget=clipped_soh,
        )

    def _apply_soh_override(self, level: ComplexityLevel, soh_budget: float) -> ComplexityLevel:
        if soh_budget >= self.SOH_DOWNGRADE_THRESHOLD or level == ComplexityLevel.LOW:
            return level
        return ComplexityLevel(level - 1)

    @staticmethod
    def _coerce_level(complexity_level: ComplexityLevel | int) -> ComplexityLevel:
        try:
            return ComplexityLevel(complexity_level)
        except ValueError as exc:
            raise ValueError("complexity_level must be one of LOW=0, MED=1, HIGH=2, CRITICAL=3") from exc
