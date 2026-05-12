from __future__ import annotations

import torch
from torch import nn


class GruPathway(nn.Module):
    """Short-term temporal reasoning pathway for entity feature sequences."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, sequence_features: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_features.ndim != 3 or sequence_features.shape[-1] != 128:
            raise ValueError("expected GRU input shape [B, T, 128]")
        _, final_hidden = self.gru(sequence_features, hidden)
        return final_hidden[-1]
