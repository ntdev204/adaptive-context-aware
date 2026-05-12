from __future__ import annotations

import pytest
import torch

from models.gated_fusion import GatedFusion


def _pathway_outputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "gru": torch.rand(batch_size, 64),
        "tcn": torch.rand(batch_size, 64),
        "attention": torch.rand(batch_size, 128),
        "gnn": torch.rand(batch_size, 256),
    }


def test_gated_fusion_forward_contract() -> None:
    model = GatedFusion()
    output = model(_pathway_outputs(batch_size=4))

    assert output.shape == (4, 256)


def test_gated_fusion_returns_interpretable_gates() -> None:
    model = GatedFusion()
    output, gates = model.forward_with_gates(_pathway_outputs(batch_size=2))

    assert output.shape == (2, 256)
    assert gates.shape == (2, 4)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_gated_fusion_supports_variable_active_pathways() -> None:
    model = GatedFusion()

    gru_output, gru_gates = model.forward_with_gates({"gru": torch.rand(2, 64)})
    mid_output, mid_gates = model.forward_with_gates(
        {
            "gru": torch.rand(2, 64),
            "tcn": torch.rand(2, 64),
            "attention": torch.rand(2, 128),
        }
    )

    assert gru_output.shape == (2, 256)
    assert gru_gates.shape == (2, 1)
    assert torch.allclose(gru_gates, torch.ones_like(gru_gates))
    assert mid_output.shape == (2, 256)
    assert mid_gates.shape == (2, 3)
    assert torch.allclose(mid_gates.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_gated_fusion_accepts_concatenated_active_tensor() -> None:
    model = GatedFusion()
    active_outputs = torch.rand(2, 64 + 128 + 256)

    output, gates = model.forward_with_gates(active_outputs, active_pathways=("gru", "attention", "gnn"))

    assert output.shape == (2, 256)
    assert gates.shape == (2, 3)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_gated_fusion_rejects_invalid_inputs() -> None:
    model = GatedFusion()

    with pytest.raises(ValueError):
        model({})
    with pytest.raises(ValueError):
        model({"gru": torch.zeros(2, 63)})
    with pytest.raises(ValueError):
        model({"gru": torch.zeros(2, 64), "tcn": torch.zeros(3, 64)})
    with pytest.raises(ValueError):
        model({"unknown": torch.zeros(2, 16)})
    with pytest.raises(ValueError):
        model(torch.zeros(2, 64))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 64), active_pathways=("gru", "gru"))


def test_gated_fusion_supports_gradient_flow() -> None:
    model = GatedFusion()
    pathway_outputs = {
        "gru": torch.rand(2, 64, requires_grad=True),
        "tcn": torch.rand(2, 64, requires_grad=True),
        "attention": torch.rand(2, 128, requires_grad=True),
        "gnn": torch.rand(2, 256, requires_grad=True),
    }

    loss = model(pathway_outputs).sum()
    loss.backward()

    for output in pathway_outputs.values():
        assert output.grad is not None
        assert torch.isfinite(output.grad).all()
