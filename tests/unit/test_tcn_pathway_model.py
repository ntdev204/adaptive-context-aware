from __future__ import annotations

import pytest
import torch

from models.tcn_pathway import TcnPathway


def test_tcn_pathway_forward_contract() -> None:
    model = TcnPathway()
    output = model(torch.zeros(4, 128, 16))

    assert output.shape == (4, 64)


def test_tcn_pathway_rejects_wrong_shape() -> None:
    model = TcnPathway()

    with pytest.raises(ValueError):
        model(torch.zeros(4, 127, 16))


def test_tcn_pathway_supports_gradient_flow() -> None:
    model = TcnPathway()
    inputs = torch.rand(3, 128, 16, requires_grad=True)

    loss = model(inputs).sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_tcn_pathway_is_causal_for_earlier_timesteps() -> None:
    model = TcnPathway()
    model.eval()
    baseline = torch.rand(1, 128, 16)
    changed_future = baseline.clone()
    changed_future[..., -1] = changed_future[..., -1] + 10.0

    with torch.no_grad():
        baseline_sequence = model.forward_sequence(baseline)
        changed_sequence = model.forward_sequence(changed_future)

    assert torch.allclose(baseline_sequence[..., :-1], changed_sequence[..., :-1], atol=1e-5)
