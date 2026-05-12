from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import GatedFusion


@dataclass(frozen=True, slots=True)
class FusionTrainingConfig:
    samples: int = 96
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 39
    checkpoint_path: Path = Path("models/checkpoints/gated_fusion.pt")


@dataclass(frozen=True, slots=True)
class FusionTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


class FusionBehaviorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = GatedFusion()
        self.activity_head = nn.Linear(256, 4)
        self.direction_head = nn.Linear(256, 9)

    def forward(self, pathway_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused = self.fusion(pathway_outputs)
        return {
            "fused": fused,
            "activity_logits": self.activity_head(fused),
            "direction_logits": self.direction_head(fused),
        }


def generate_synthetic_fusion_dataset(samples: int = 96, seed: int = 39) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")

    generator = torch.Generator().manual_seed(seed)
    activity_labels = torch.arange(samples, dtype=torch.long) % 4
    direction_labels = torch.arange(samples, dtype=torch.long) % 9
    gru = torch.randn(samples, 64, generator=generator) * 0.05
    tcn = torch.randn(samples, 64, generator=generator) * 0.05
    attention = torch.randn(samples, 128, generator=generator) * 0.05
    gnn = torch.randn(samples, 256, generator=generator) * 0.05
    gru[:, 0] = activity_labels.float() / 3.0
    tcn[:, 0] = direction_labels.float() / 8.0
    attention[:, 0] = activity_labels.float() / 3.0
    gnn[:, 0] = direction_labels.float() / 8.0
    return TensorDataset(gru, tcn, attention, gnn, activity_labels, direction_labels)


def compute_fusion_loss(model: FusionBehaviorHead, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    gru, tcn, attention, gnn, activity_labels, direction_labels = batch
    outputs = model({"gru": gru, "tcn": tcn, "attention": attention, "gnn": gnn})
    activity_loss = nn.functional.cross_entropy(outputs["activity_logits"], activity_labels)
    direction_loss = nn.functional.cross_entropy(outputs["direction_logits"], direction_labels)
    return activity_loss + direction_loss


def train_fusion(config: FusionTrainingConfig = FusionTrainingConfig()) -> FusionTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_fusion_dataset(samples=config.samples, seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = FusionBehaviorHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_fusion_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return FusionTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def _save_checkpoint(model: FusionBehaviorHead, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gated fusion layer on synthetic pathway outputs.")
    parser.add_argument("--samples", type=int, default=FusionTrainingConfig.samples)
    parser.add_argument("--epochs", type=int, default=FusionTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=FusionTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=FusionTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=FusionTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_fusion(
        FusionTrainingConfig(
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
