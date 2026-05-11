from __future__ import annotations

import pytest
import torch

from models.complexity_estimator import ComplexityEstimatorNet


def test_complexity_estimator_forward_contract() -> None:
    model = ComplexityEstimatorNet()
    output = model(torch.zeros(8, 36))

    assert output.shape == (8, 4)


def test_complexity_estimator_rejects_wrong_shape() -> None:
    model = ComplexityEstimatorNet()

    with pytest.raises(ValueError):
        model(torch.zeros(8, 35))


def test_complexity_estimator_supports_gradient_flow() -> None:
    model = ComplexityEstimatorNet()
    inputs = torch.rand(8, 36, requires_grad=True)

    loss = model(inputs).sum()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_complexity_estimator_can_learn_synthetic_complexity_cases() -> None:
    torch.manual_seed(7)
    model = ComplexityEstimatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()
    inputs, labels = _synthetic_cases()

    model.train()
    for _ in range(120):
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(inputs), dim=1)

    assert torch.equal(predictions, labels)


def _synthetic_cases() -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    for level, base in enumerate((0.05, 0.35, 0.65, 0.90)):
        for offset in (0.0, 0.03):
            row = torch.zeros(36)
            row[0] = base + offset
            row[1] = base
            row[2] = base
            row[3] = 1.0 - base
            row[4:] = base
            rows.append(row)
            labels.append(level)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.long)
