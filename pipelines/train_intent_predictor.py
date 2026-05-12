from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import IntentPredictorNet


@dataclass(frozen=True, slots=True)
class IntentPredictorTrainingConfig:
    samples: int = 96
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 41
    checkpoint_path: Path = Path("models/checkpoints/intent_predictor.pt")


@dataclass(frozen=True, slots=True)
class IntentPredictorTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


def generate_synthetic_intent_dataset(samples: int = 96, seed: int = 41) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")

    generator = torch.Generator().manual_seed(seed)
    fused = torch.randn(samples, 256, generator=generator) * 0.05
    direction_labels = torch.arange(samples, dtype=torch.long) % 9
    activity_labels = torch.arange(samples, dtype=torch.long) % 9
    trajectory_targets = torch.randn(samples, 2, 2, generator=generator) * 0.1
    fused[:, 0] = direction_labels.float() / 8.0
    fused[:, 1] = activity_labels.float() / 8.0
    return TensorDataset(fused, direction_labels, activity_labels, trajectory_targets)


def compute_intent_predictor_loss(model: IntentPredictorNet, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    fused, direction_labels, activity_labels, trajectory_targets = batch
    outputs = model(fused)
    direction_loss = nn.functional.cross_entropy(outputs["direction_logits"], direction_labels)
    activity_loss = nn.functional.cross_entropy(outputs["activity_logits"], activity_labels)
    trajectory_loss = nn.functional.mse_loss(outputs["trajectory_offsets"], trajectory_targets)
    return direction_loss + activity_loss + 0.2 * trajectory_loss


def train_intent_predictor(
    config: IntentPredictorTrainingConfig = IntentPredictorTrainingConfig(),
) -> IntentPredictorTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_intent_dataset(samples=config.samples, seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = IntentPredictorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_intent_predictor_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return IntentPredictorTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def _save_checkpoint(model: IntentPredictorNet, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the intent predictor on synthetic fused reasoning outputs.")
    parser.add_argument("--samples", type=int, default=IntentPredictorTrainingConfig.samples)
    parser.add_argument("--epochs", type=int, default=IntentPredictorTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=IntentPredictorTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=IntentPredictorTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=IntentPredictorTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_intent_predictor(
        IntentPredictorTrainingConfig(
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
