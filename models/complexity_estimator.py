from __future__ import annotations

import torch
from torch import nn


class ComplexityEstimatorNet(nn.Module):
    """Tiny gatekeeper MLP for Phase 2 routing complexity."""

    def __init__(self, input_dim: int = 36, hidden_dims: tuple[int, int] = (64, 32), output_dim: int = 4) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.BatchNorm1d(first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.BatchNorm1d(second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 36:
            raise ValueError("expected complexity features with shape [B, 36]")
        return self.network(features)
