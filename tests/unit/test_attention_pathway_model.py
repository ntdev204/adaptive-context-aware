from __future__ import annotations

import pytest
import torch

from models.attention_pathway import AttentionPathway


def test_attention_pathway_forward_contract() -> None:
    model = AttentionPathway()
    output = model(torch.zeros(3, 6, 128))

    assert output.shape == (3, 128)


def test_attention_pathway_returns_interpretable_attention_weights() -> None:
    model = AttentionPathway()
    _, weights = model.forward_with_attention(torch.rand(2, 5, 128))

    assert weights.shape == (2, 4, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-5)


def test_attention_pathway_supports_variable_entity_counts() -> None:
    model = AttentionPathway()

    assert model(torch.rand(2, 1, 128)).shape == (2, 128)
    assert model(torch.rand(2, 9, 128)).shape == (2, 128)


def test_attention_pathway_rejects_wrong_shape() -> None:
    model = AttentionPathway()

    with pytest.raises(ValueError):
        model(torch.zeros(3, 6, 127))


def test_attention_pathway_rejects_empty_entity_axis() -> None:
    model = AttentionPathway()

    with pytest.raises(ValueError):
        model(torch.zeros(3, 0, 128))


def test_attention_pathway_supports_gradient_flow() -> None:
    model = AttentionPathway()
    inputs = torch.rand(2, 6, 128, requires_grad=True)

    loss = model(inputs).sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
