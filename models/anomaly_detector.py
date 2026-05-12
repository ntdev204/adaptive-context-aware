from __future__ import annotations

import torch
from torch import nn


class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 256, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 160),
            nn.ReLU(),
            nn.Linear(160, input_dim),
        )

    def forward(self, fused_reasoning: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(fused_reasoning)
        return self.decoder(latent)
