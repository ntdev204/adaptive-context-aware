from __future__ import annotations

import torch
from torch import nn


class _CausalTrim1d(nn.Module):
    def __init__(self, trim_size: int) -> None:
        super().__init__()
        self.trim_size = trim_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.trim_size == 0:
            return values
        return values[..., : -self.trim_size]


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (3 - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding),
            _CausalTrim1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(values) + self.residual(values))


class TcnPathway(nn.Module):
    """Longer-horizon temporal pathway using causal dilated convolutions."""

    def __init__(self, input_channels: int = 128, hidden_channels: tuple[int, int, int] = (128, 64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for dilation, out_channels in zip((1, 2, 4), hidden_channels, strict=True):
            layers.append(_TemporalBlock(in_channels, out_channels, dilation=dilation))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        if sequence_features.ndim != 3 or sequence_features.shape[1] != 128:
            raise ValueError("expected TCN input shape [B, 128, T]")
        return self.forward_sequence(sequence_features)[..., -1]

    def forward_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        if sequence_features.ndim != 3 or sequence_features.shape[1] != 128:
            raise ValueError("expected TCN input shape [B, 128, T]")
        return self.network(sequence_features)
