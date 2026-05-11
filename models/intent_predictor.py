from __future__ import annotations

import torch
from torch import nn


class IntentPredictorNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 192,
        direction_classes: int = 9,
        activity_classes: int = 9,
        trajectory_steps: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.direction_classes = direction_classes
        self.activity_classes = activity_classes
        self.trajectory_steps = trajectory_steps

        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(hidden_dim, direction_classes)
        self.activity_head = nn.Linear(hidden_dim, activity_classes)
        self.trajectory_head = nn.Linear(hidden_dim, trajectory_steps * 2)

    def forward(self, fused_reasoning: torch.Tensor) -> dict[str, torch.Tensor]:
        if fused_reasoning.ndim != 2 or fused_reasoning.shape[-1] != self.input_dim:
            raise ValueError(f"expected fused_reasoning shape [B, {self.input_dim}]")

        hidden = self.backbone(fused_reasoning)
        trajectory = self.trajectory_head(hidden).reshape(-1, self.trajectory_steps, 2)
        return {
            "direction_logits": self.direction_head(hidden),
            "activity_logits": self.activity_head(hidden),
            "trajectory_offsets": trajectory,
        }
