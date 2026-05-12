from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import TcnPathway


@dataclass(frozen=True, slots=True)
class TcnTrainingConfig:
    samples: int = 128
    sequence_length: int = 8
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 33
    checkpoint_path: Path = Path("models/checkpoints/tcn_pathway.pt")


@dataclass(frozen=True, slots=True)
class TcnTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


class TcnBehaviorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tcn = TcnPathway()
        self.direction_head = nn.Linear(64, 9)
        self.speed_head = nn.Linear(64, 1)

    def forward(self, sequence_features: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = self.tcn(sequence_features)
        return {
            "summary": summary,
            "direction_logits": self.direction_head(summary),
            "speed": self.speed_head(summary).squeeze(-1),
        }


def generate_synthetic_tcn_dataset(
    samples: int = 128,
    sequence_length: int = 8,
    seed: int = 33,
) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    generator = torch.Generator().manual_seed(seed)
    direction_labels = torch.arange(samples, dtype=torch.long) % 9
    speed_targets = 0.2 + (direction_labels.float() / 8.0) * 1.8
    sequence_features = torch.randn(samples, 128, sequence_length, generator=generator) * 0.05
    sequence_features[:, 0, :] = direction_labels.float().unsqueeze(1) / 8.0
    sequence_features[:, 1, :] = speed_targets.unsqueeze(1)
    return TensorDataset(sequence_features, direction_labels, speed_targets)


def compute_tcn_loss(model: TcnBehaviorHead, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    sequence_features, direction_labels, speed_targets = batch
    outputs = model(sequence_features)
    direction_loss = nn.functional.cross_entropy(outputs["direction_logits"], direction_labels)
    speed_loss = nn.functional.mse_loss(outputs["speed"], speed_targets)
    return direction_loss + 0.2 * speed_loss


def train_tcn(config: TcnTrainingConfig = TcnTrainingConfig()) -> TcnTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_tcn_dataset(
        samples=config.samples,
        sequence_length=config.sequence_length,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = TcnBehaviorHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_tcn_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return TcnTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def _save_checkpoint(model: TcnBehaviorHead, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TCN pathway on synthetic temporal behavior data.")
    parser.add_argument("--samples", type=int, default=TcnTrainingConfig.samples)
    parser.add_argument("--sequence-length", type=int, default=TcnTrainingConfig.sequence_length)
    parser.add_argument("--epochs", type=int, default=TcnTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TcnTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TcnTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=TcnTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_tcn(
        TcnTrainingConfig(
            samples=args.samples,
            sequence_length=args.sequence_length,
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
