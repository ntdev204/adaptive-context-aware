from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import AttentionPathway, GatedFusion, GraphAttentionPathway, GruPathway, TcnPathway


@dataclass(frozen=True, slots=True)
class ReasoningTrainingConfig:
    samples: int = 96
    sequence_length: int = 8
    entity_count: int = 6
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 23
    checkpoint_path: Path = Path("models/checkpoints/reasoning_brain.pt")


@dataclass(frozen=True, slots=True)
class ReasoningTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


class JointReasoningNet(nn.Module):
    def __init__(self, activity_classes: int = 4, direction_classes: int = 9) -> None:
        super().__init__()
        self.gru = GruPathway()
        self.tcn = TcnPathway()
        self.attention = AttentionPathway()
        self.gnn = GraphAttentionPathway()
        self.fusion = GatedFusion()
        self.activity_head = nn.Linear(256, activity_classes)
        self.direction_head = nn.Linear(256, direction_classes)
        self.anomaly_head = nn.Linear(256, 1)
        self.reconstruction_head = nn.Linear(256, 128)

    def forward(
        self,
        sequence_features: torch.Tensor,
        entity_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gru_output = self.gru(sequence_features)
        tcn_output = self.tcn(sequence_features.transpose(1, 2))
        attention_output = self.attention(entity_features)
        gnn_output = self.gnn(entity_features, adjacency)
        fused = self.fusion(
            {
                "gru": gru_output,
                "tcn": tcn_output,
                "attention": attention_output,
                "gnn": gnn_output,
            }
        )
        return {
            "fused": fused,
            "activity_logits": self.activity_head(fused),
            "direction_logits": self.direction_head(fused),
            "anomaly_logits": self.anomaly_head(fused).squeeze(-1),
            "reconstruction": self.reconstruction_head(fused),
        }


def generate_synthetic_reasoning_dataset(
    samples: int = 96,
    sequence_length: int = 8,
    entity_count: int = 6,
    seed: int = 23,
) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sequence_length <= 0 or entity_count <= 0:
        raise ValueError("sequence_length and entity_count must be positive")

    generator = torch.Generator().manual_seed(seed)
    activity_labels = torch.arange(samples, dtype=torch.long) % 4
    direction_labels = torch.arange(samples, dtype=torch.long) % 9
    anomaly_labels = (activity_labels == 3).to(dtype=torch.float32)

    sequence_features = torch.randn(samples, sequence_length, 128, generator=generator) * 0.04
    entity_features = torch.randn(samples, entity_count, 128, generator=generator) * 0.04

    activity_signal = activity_labels.float() / 3.0
    direction_signal = direction_labels.float() / 8.0
    sequence_features[:, :, 0] = activity_signal.unsqueeze(1)
    sequence_features[:, :, 1] = direction_signal.unsqueeze(1)
    sequence_features[:, :, 2] = anomaly_labels.unsqueeze(1)
    entity_features[:, :, 0] = activity_signal.unsqueeze(1)
    entity_features[:, :, 1] = direction_signal.unsqueeze(1)
    entity_features[:, :, 2] = anomaly_labels.unsqueeze(1)

    adjacency = torch.ones(samples, entity_count, entity_count, dtype=torch.float32)
    reconstruction_targets = entity_features.mean(dim=1)
    return TensorDataset(
        sequence_features,
        entity_features,
        adjacency,
        activity_labels,
        direction_labels,
        anomaly_labels,
        reconstruction_targets,
    )


def train_reasoning(config: ReasoningTrainingConfig = ReasoningTrainingConfig()) -> ReasoningTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_reasoning_dataset(
        samples=config.samples,
        sequence_length=config.sequence_length,
        entity_count=config.entity_count,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = JointReasoningNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_reasoning_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return ReasoningTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def compute_reasoning_loss(model: JointReasoningNet, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    (
        sequence_features,
        entity_features,
        adjacency,
        activity_labels,
        direction_labels,
        anomaly_labels,
        reconstruction_targets,
    ) = batch
    outputs = model(sequence_features, entity_features, adjacency)
    activity_loss = nn.functional.cross_entropy(outputs["activity_logits"], activity_labels)
    direction_loss = nn.functional.cross_entropy(outputs["direction_logits"], direction_labels)
    anomaly_loss = nn.functional.binary_cross_entropy_with_logits(outputs["anomaly_logits"], anomaly_labels)
    reconstruction_loss = nn.functional.mse_loss(outputs["reconstruction"], reconstruction_targets)
    return activity_loss + direction_loss + anomaly_loss + 0.1 * reconstruction_loss


def _save_checkpoint(model: JointReasoningNet, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "activity_classes": 4,
            "direction_classes": 9,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint-train Phase 2 reasoning pathways and fusion.")
    parser.add_argument("--samples", type=int, default=ReasoningTrainingConfig.samples)
    parser.add_argument("--epochs", type=int, default=ReasoningTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ReasoningTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ReasoningTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=ReasoningTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_reasoning(
        ReasoningTrainingConfig(
            samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_path=args.checkpoint_path,
        )
    )
    print(f"final_loss={result.final_loss:.4f}")
    print(f"checkpoint_path={result.checkpoint_path}")


if __name__ == "__main__":
    main()
