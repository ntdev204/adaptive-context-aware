from __future__ import annotations

import pytest

from src.complexity.estimator import ComplexityLevel
from src.router.adaptive_router import AdaptiveRouter, ReasoningPathway


def test_adaptive_router_low_uses_gru_only() -> None:
    decision = AdaptiveRouter().route(ComplexityLevel.LOW, soh_budget=1.0)

    assert decision.active_pathways == (ReasoningPathway.GRU,)
    assert decision.latency_budget_ms == pytest.approx(5.0)
    assert not decision.downgraded


def test_adaptive_router_high_uses_gru_tcn_attention() -> None:
    decision = AdaptiveRouter().route(ComplexityLevel.HIGH, soh_budget=1.0)

    assert decision.active_pathways == (
        ReasoningPathway.GRU,
        ReasoningPathway.TCN,
        ReasoningPathway.ATTENTION,
    )
    assert decision.latency_budget_ms == pytest.approx(20.0)


def test_adaptive_router_critical_uses_all_pathways() -> None:
    decision = AdaptiveRouter().route(ComplexityLevel.CRITICAL, soh_budget=1.0)

    assert decision.active_pathways == (
        ReasoningPathway.GRU,
        ReasoningPathway.TCN,
        ReasoningPathway.ATTENTION,
        ReasoningPathway.GNN,
    )
    assert sum(decision.pathway_budget_ms.values()) <= decision.latency_budget_ms


def test_adaptive_router_soh_override_downgrades_one_level() -> None:
    decision = AdaptiveRouter().route(ComplexityLevel.HIGH, soh_budget=0.49)

    assert decision.requested_level == ComplexityLevel.HIGH
    assert decision.effective_level == ComplexityLevel.MED
    assert decision.active_pathways == (ReasoningPathway.GRU, ReasoningPathway.TCN)
    assert decision.downgraded


def test_adaptive_router_clips_soh_budget_and_rejects_invalid_level() -> None:
    router = AdaptiveRouter()

    assert router.route(ComplexityLevel.LOW, soh_budget=2.0).soh_budget == 1.0
    assert router.route(ComplexityLevel.LOW, soh_budget=-1.0).soh_budget == 0.0
    with pytest.raises(ValueError):
        router.route(4, soh_budget=1.0)
