from __future__ import annotations

import pytest
import torch

from models.gnn_pathway import GraphAttentionPathway


def test_gnn_pathway_forward_contract() -> None:
    model = GraphAttentionPathway()
    output = model(torch.zeros(3, 6, 128), torch.ones(3, 6, 6))

    assert output.shape == (3, 256)


def test_gnn_pathway_returns_attention_weights() -> None:
    model = GraphAttentionPathway()
    _, (layer1_weights, layer2_weights) = model.forward_with_attention(torch.rand(2, 5, 128), torch.ones(2, 5, 5))

    assert layer1_weights.shape == (2, 4, 5, 5)
    assert layer2_weights.shape == (2, 1, 5, 5)
    assert torch.allclose(layer1_weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-5)
    assert torch.allclose(layer2_weights.sum(dim=-1), torch.ones(2, 1, 5), atol=1e-5)


def test_gnn_pathway_builds_distance_based_adjacency() -> None:
    positions = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ]
    )

    adjacency = GraphAttentionPathway.build_adjacency(positions, threshold_m=2.0)

    assert adjacency.shape == (1, 3, 3)
    assert adjacency[0, 0, 0] == 1
    assert adjacency[0, 0, 1] == 1
    assert adjacency[0, 0, 2] == 0


def test_gnn_pathway_supports_unbatched_adjacency() -> None:
    model = GraphAttentionPathway()
    output = model(torch.rand(2, 4, 128), torch.ones(4, 4))

    assert output.shape == (2, 256)


def test_gnn_pathway_handles_empty_entities() -> None:
    model = GraphAttentionPathway()
    output = model(torch.zeros(2, 0, 128), torch.zeros(2, 0, 0))

    assert output.shape == (2, 256)
    assert torch.count_nonzero(output) == 0


def test_gnn_pathway_rejects_wrong_shapes() -> None:
    model = GraphAttentionPathway()

    with pytest.raises(ValueError):
        model(torch.zeros(2, 4, 127), torch.ones(2, 4, 4))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 4, 128), torch.ones(2, 3, 3))
    with pytest.raises(ValueError):
        GraphAttentionPathway.build_adjacency(torch.zeros(2, 4, 2))


def test_gnn_pathway_supports_gradient_flow() -> None:
    model = GraphAttentionPathway()
    inputs = torch.rand(2, 6, 128, requires_grad=True)

    loss = model(inputs, torch.ones(2, 6, 6)).sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
