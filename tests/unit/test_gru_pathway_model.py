from __future__ import annotations

import pytest
import torch

from models.gru_pathway import GruPathway


def test_gru_pathway_forward_contract() -> None:
    model = GruPathway()
    output = model(torch.zeros(4, 8, 128))

    assert output.shape == (4, 64)


def test_gru_pathway_rejects_wrong_shape() -> None:
    model = GruPathway()

    with pytest.raises(ValueError):
        model(torch.zeros(4, 8, 127))


def test_gru_pathway_supports_gradient_flow() -> None:
    model = GruPathway()
    inputs = torch.rand(3, 8, 128, requires_grad=True)

    loss = model(inputs).sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_gru_pathway_accepts_stateful_hidden_input() -> None:
    model = GruPathway()
    inputs = torch.rand(2, 8, 128)
    hidden = torch.zeros(1, 2, 64)

    output = model(inputs, hidden=hidden)

    assert output.shape == (2, 64)
