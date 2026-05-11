from __future__ import annotations

import torch

from models import (
    AttentionPathway,
    ComplexityEstimatorNet,
    GatedFusion,
    GraphAttentionPathway,
    GruPathway,
    TcnPathway,
)


def test_brain_network_forward_shapes() -> None:
    assert ComplexityEstimatorNet()(torch.zeros(2, 36)).shape == (2, 4)
    assert GruPathway()(torch.zeros(2, 8, 128)).shape == (2, 64)
    assert TcnPathway()(torch.zeros(2, 128, 8)).shape == (2, 64)
    assert AttentionPathway()(torch.zeros(2, 3, 128)).shape == (2, 128)
    assert GraphAttentionPathway()(torch.zeros(2, 3, 128), torch.ones(2, 3, 3)).shape == (2, 256)
    assert GatedFusion()(
        {
            "gru": torch.zeros(2, 64),
            "tcn": torch.zeros(2, 64),
            "attention": torch.zeros(2, 128),
            "gnn": torch.zeros(2, 256),
        }
    ).shape == (2, 256)


def test_brain_network_gradient_flow_smoke() -> None:
    complexity_inputs = torch.rand(2, 36, requires_grad=True)
    sequence_inputs = torch.rand(2, 8, 128, requires_grad=True)
    entity_inputs = torch.rand(2, 4, 128, requires_grad=True)

    loss = ComplexityEstimatorNet()(complexity_inputs).sum()
    loss = loss + GruPathway()(sequence_inputs).sum()
    loss = loss + TcnPathway()(sequence_inputs.transpose(1, 2)).sum()
    loss = loss + AttentionPathway()(entity_inputs).sum()
    loss = loss + GraphAttentionPathway()(entity_inputs, torch.ones(2, 4, 4)).sum()
    loss.backward()

    assert complexity_inputs.grad is not None
    assert sequence_inputs.grad is not None
    assert entity_inputs.grad is not None


def test_brain_network_edge_cases_t1_single_entity_and_empty_graph() -> None:
    assert GruPathway()(torch.zeros(2, 1, 128)).shape == (2, 64)
    assert TcnPathway()(torch.zeros(2, 128, 1)).shape == (2, 64)
    assert AttentionPathway()(torch.zeros(2, 1, 128)).shape == (2, 128)
    assert GraphAttentionPathway()(torch.zeros(2, 1, 128), torch.ones(2, 1, 1)).shape == (2, 256)
    assert GraphAttentionPathway()(torch.zeros(2, 0, 128), torch.zeros(2, 0, 0)).shape == (2, 256)


def test_gated_fusion_single_and_partial_active_pathways() -> None:
    fusion = GatedFusion()

    gru_only, gru_gates = fusion.forward_with_gates({"gru": torch.zeros(2, 64)})
    temporal_only, temporal_gates = fusion.forward_with_gates(
        {
            "gru": torch.zeros(2, 64),
            "tcn": torch.zeros(2, 64),
        }
    )

    assert gru_only.shape == (2, 256)
    assert gru_gates.shape == (2, 1)
    assert torch.allclose(gru_gates, torch.ones_like(gru_gates))
    assert temporal_only.shape == (2, 256)
    assert temporal_gates.shape == (2, 2)
    assert torch.allclose(temporal_gates.sum(dim=-1), torch.ones(2), atol=1e-6)
