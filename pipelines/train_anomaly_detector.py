from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import AnomalyAutoencoder


@dataclass(frozen=True, slots=True)
class AnomalyDetectorTrainingConfig:
    samples: int = 96
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 43
    checkpoint_path: Path = Path("models/checkpoints/anomaly_detector.pt")


@dataclass(frozen=True, slots=True)
class AnomalyDetectorTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


def generate_synthetic_anomaly_dataset(samples: int = 96, seed: int = 43) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")

    generator = torch.Generator().manual_seed(seed)
    fused = torch.randn(samples, 256, generator=generator) * 0.05
    anomaly_labels = (torch.arange(samples) % 4 == 3).float()
    fused[:, 0] = anomaly_labels
    return TensorDataset(fused, anomaly_labels)


def compute_anomaly_detector_loss(model: AnomalyAutoencoder, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    fused, anomaly_labels = batch
    reconstructed = model(fused)
    reconstruction_loss = nn.functional.mse_loss(reconstructed, fused)
    anomaly_weight = 1.0 + anomaly_labels.unsqueeze(1)
    weighted_loss = ((reconstructed - fused) ** 2 * anomaly_weight).mean()
    return reconstruction_loss + weighted_loss


def train_anomaly_detector(
    config: AnomalyDetectorTrainingConfig = AnomalyDetectorTrainingConfig(),
) -> AnomalyDetectorTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_anomaly_dataset(samples=config.samples, seed=config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = AnomalyAutoencoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_anomaly_detector_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return AnomalyDetectorTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def _save_checkpoint(model: AnomalyAutoencoder, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the anomaly autoencoder on synthetic fused reasoning outputs.")
    parser.add_argument("--samples", type=int, default=AnomalyDetectorTrainingConfig.samples)
    parser.add_argument("--epochs", type=int, default=AnomalyDetectorTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=AnomalyDetectorTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=AnomalyDetectorTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=AnomalyDetectorTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_anomaly_detector(
        AnomalyDetectorTrainingConfig(
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
