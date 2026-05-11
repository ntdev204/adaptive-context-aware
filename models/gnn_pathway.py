from __future__ import annotations

import torch
from torch import nn


class _GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, num_heads * output_dim, bias=False)
        self.key = nn.Linear(input_dim, num_heads * output_dim, bias=False)
        self.value = nn.Linear(input_dim, num_heads * output_dim, bias=False)
        self.output = nn.Linear(num_heads * output_dim, output_dim)

    def forward(self, entity_features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, entity_count, _ = entity_features.shape
        query = self._project(self.query(entity_features), batch_size, entity_count)
        key = self._project(self.key(entity_features), batch_size, entity_count)
        value = self._project(self.value(entity_features), batch_size, entity_count)

        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.output_dim**0.5)
        scores = scores.masked_fill(~adjacency.unsqueeze(1).bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch_size, entity_count, self.num_heads * self.output_dim)
        return self.output(context), weights

    def _project(self, values: torch.Tensor, batch_size: int, entity_count: int) -> torch.Tensor:
        return values.view(batch_size, entity_count, self.num_heads, self.output_dim).transpose(1, 2)


class GraphAttentionPathway(nn.Module):
    """Spatial graph attention pathway for entity interaction reasoning."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = _GraphAttentionLayer(input_dim=input_dim, output_dim=hidden_dim, num_heads=4)
        self.layer2 = _GraphAttentionLayer(input_dim=hidden_dim, output_dim=output_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, entity_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        context, _ = self.forward_with_attention(entity_features, adjacency)
        return context

    def forward_with_attention(
        self,
        entity_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        self._validate_entity_features(entity_features)
        adjacency = self._normalize_adjacency(entity_features, adjacency)
        batch_size, entity_count, _ = entity_features.shape
        if entity_count == 0:
            empty_weights = entity_features.new_zeros(batch_size, 0, 0, 0)
            return entity_features.new_zeros(batch_size, self.output_dim), (empty_weights, empty_weights)

        hidden, layer1_weights = self.layer1(entity_features, adjacency)
        hidden = self.activation(self.norm1(hidden))
        output, layer2_weights = self.layer2(hidden, adjacency)
        output = self.activation(self.norm2(output))
        return output.mean(dim=1), (layer1_weights, layer2_weights)

    @staticmethod
    def build_adjacency(positions_xyz_m: torch.Tensor, threshold_m: float = 2.0) -> torch.Tensor:
        if positions_xyz_m.ndim == 2:
            positions_xyz_m = positions_xyz_m.unsqueeze(0)
        if positions_xyz_m.ndim != 3 or positions_xyz_m.shape[-1] != 3:
            raise ValueError("expected positions shape [B, N, 3] or [N, 3]")
        batch_size, entity_count, _ = positions_xyz_m.shape
        if entity_count == 0:
            return positions_xyz_m.new_zeros(batch_size, 0, 0)

        distances = torch.cdist(positions_xyz_m, positions_xyz_m, p=2)
        adjacency = (distances <= threshold_m).to(dtype=positions_xyz_m.dtype)
        self_loops = torch.eye(entity_count, device=positions_xyz_m.device, dtype=positions_xyz_m.dtype)
        return torch.maximum(adjacency, self_loops.unsqueeze(0))

    def _validate_entity_features(self, entity_features: torch.Tensor) -> None:
        if entity_features.ndim != 3 or entity_features.shape[-1] != self.input_dim:
            raise ValueError("expected GNN input shape [B, N, 128]")

    @staticmethod
    def _normalize_adjacency(entity_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0).expand(entity_features.shape[0], -1, -1)
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("expected adjacency shape [B, N, N] or [N, N]")
        if adjacency.shape[0] != entity_features.shape[0] or adjacency.shape[1] != entity_features.shape[1]:
            raise ValueError("adjacency must align with GNN entity tensor")

        entity_count = adjacency.shape[-1]
        if entity_count == 0:
            return adjacency.to(device=entity_features.device, dtype=entity_features.dtype)
        self_loops = torch.eye(entity_count, device=entity_features.device, dtype=entity_features.dtype)
        adjacency = adjacency.to(device=entity_features.device, dtype=entity_features.dtype)
        return torch.maximum(adjacency, self_loops.unsqueeze(0))
