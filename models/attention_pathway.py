from __future__ import annotations

import torch
from torch import nn


class AttentionPathway(nn.Module):
    """Inter-entity self-attention pathway for Phase 2 reasoning."""

    def __init__(self, input_dim: int = 128, num_heads: int = 4, feedforward_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, input_dim),
        )

    def forward(self, entity_features: torch.Tensor) -> torch.Tensor:
        context, _ = self.forward_with_attention(entity_features)
        return context

    def forward_with_attention(self, entity_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if entity_features.ndim != 3 or entity_features.shape[-1] != self.input_dim:
            raise ValueError("expected attention input shape [B, N, 128]")
        if entity_features.shape[1] <= 0:
            raise ValueError("attention pathway requires at least one entity")

        normalized = self.input_norm(entity_features)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = entity_features + attended
        hidden = hidden + self.feedforward(self.output_norm(hidden))
        return hidden.mean(dim=1), weights
