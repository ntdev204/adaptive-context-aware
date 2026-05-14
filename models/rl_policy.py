from __future__ import annotations

import torch
from torch import nn


class RLPolicyNet(nn.Module):
    def __init__(self, input_dim: int = 39, hidden_dim: int = 64, output_dim: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
