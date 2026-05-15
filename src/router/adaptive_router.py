from __future__ import annotations

from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 on Jetson
    from strenum import StrEnum

import numpy as np

from src.complexity.estimator import ComplexityLevel
from src.router.rl_policy import RLPolicy, RLRouterState, RoutingAction


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
    strategy: str = "rule-based"
    fallback_used: bool = False
    selected_action: RoutingAction | None = None

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

    ACTION_TO_LEVEL = {
        RoutingAction.FAST: ComplexityLevel.LOW,
        RoutingAction.BALANCED: ComplexityLevel.MED,
        RoutingAction.ACCURATE: ComplexityLevel.HIGH,
        RoutingAction.EMERGENCY: ComplexityLevel.CRITICAL,
    }

    def __init__(self, rl_policy: RLPolicy | None = None) -> None:
        self.rl_policy = rl_policy

    def route(
        self,
        complexity_level: ComplexityLevel | int,
        soh_budget: float,
        *,
        rl_state: RLRouterState | None = None,
        use_rl: bool = True,
    ) -> RoutingDecision:
        requested_level = self._coerce_level(complexity_level)
        clipped_soh = float(np.clip(soh_budget, 0.0, 1.0))
        if use_rl and self.rl_policy is not None and rl_state is not None:
            try:
                rl_decision = self.rl_policy.decide(rl_state)
            except Exception:
                return self._route_rule_based(requested_level, clipped_soh, fallback_used=True)
            rl_level = self.ACTION_TO_LEVEL[rl_decision.action]
            return self._build_decision(
                requested_level=requested_level,
                effective_level=self._apply_soh_override(rl_level, clipped_soh),
                soh_budget=clipped_soh,
                strategy="rl-policy",
                selected_action=rl_decision.action,
            )
        return self._route_rule_based(requested_level, clipped_soh)

    def _route_rule_based(
        self,
        requested_level: ComplexityLevel,
        soh_budget: float,
        *,
        fallback_used: bool = False,
    ) -> RoutingDecision:
        return self._build_decision(
            requested_level=requested_level,
            effective_level=self._apply_soh_override(requested_level, soh_budget),
            soh_budget=soh_budget,
            strategy="rule-based",
            fallback_used=fallback_used,
        )

    def _build_decision(
        self,
        *,
        requested_level: ComplexityLevel,
        effective_level: ComplexityLevel,
        soh_budget: float,
        strategy: str,
        fallback_used: bool = False,
        selected_action: RoutingAction | None = None,
    ) -> RoutingDecision:
        active_pathways = self.ROUTING_TABLE[effective_level]
        return RoutingDecision(
            requested_level=requested_level,
            effective_level=effective_level,
            active_pathways=active_pathways,
            latency_budget_ms=self.LATENCY_BUDGET_MS[effective_level],
            pathway_budget_ms={pathway: self.PATHWAY_TARGET_MS[pathway] for pathway in active_pathways},
            soh_budget=soh_budget,
            strategy=strategy,
            fallback_used=fallback_used,
            selected_action=selected_action,
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
