from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.complexity_estimator import ComplexityEstimatorNet


@dataclass(frozen=True, slots=True)
class EstimatorTrainingConfig:
    samples_per_level: int = 128
    validation_fraction: float = 0.2
    epochs: int = 160
    batch_size: int = 32
    learning_rate: float = 0.02
    seed: int = 7
    checkpoint_path: Path = Path("models/checkpoints/complexity_estimator.pt")


@dataclass(frozen=True, slots=True)
class EstimatorTrainingResult:
    validation_accuracy: float
    final_train_loss: float
    epochs: int
    checkpoint_path: Path | None


def generate_synthetic_complexity_dataset(
    samples_per_level: int = 128,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples_per_level <= 0:
        raise ValueError("samples_per_level must be positive")

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    level_specs = (
        ((0.00, 0.20), (0.00, 0.25), (0.00, 0.10), (0.85, 1.00), 0.05),
        ((0.25, 0.45), (0.20, 0.50), (0.00, 0.20), (0.70, 1.00), 0.35),
        ((0.50, 0.75), (0.45, 0.75), (0.20, 0.50), (0.45, 0.85), 0.65),
        ((0.75, 1.00), (0.65, 1.00), (0.50, 1.00), (0.20, 0.70), 0.90),
    )

    for level, (density_band, entropy_band, anomaly_band, soh_band, embedding_center) in enumerate(level_specs):
        for _ in range(samples_per_level):
            scene_embedding = np.clip(
                rng.normal(loc=embedding_center, scale=0.04, size=32),
                0.0,
                1.0,
            ).astype(np.float32)
            row = np.concatenate(
                [
                    np.array(
                        [
                            rng.uniform(*density_band),
                            rng.uniform(*entropy_band),
                            rng.uniform(*anomaly_band),
                            rng.uniform(*soh_band),
                        ],
                        dtype=np.float32,
                    ),
                    scene_embedding,
                ]
            )
            rows.append(row)
            labels.append(level)

    features = torch.tensor(np.stack(rows), dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    return features, targets


def train_estimator(config: EstimatorTrainingConfig = EstimatorTrainingConfig()) -> EstimatorTrainingResult:
    torch.manual_seed(config.seed)
    features, labels = generate_synthetic_complexity_dataset(
        samples_per_level=config.samples_per_level,
        seed=config.seed,
    )
    train_dataset, validation_dataset = _split_dataset(features, labels, config.validation_fraction, config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = ComplexityEstimatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    validation_accuracy = evaluate_accuracy(model, validation_dataset)
    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return EstimatorTrainingResult(
        validation_accuracy=validation_accuracy,
        final_train_loss=final_loss,
        epochs=config.epochs,
        checkpoint_path=checkpoint_path,
    )


def evaluate_accuracy(model: ComplexityEstimatorNet, dataset: TensorDataset) -> float:
    loader = DataLoader(dataset, batch_size=128)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            predictions = torch.argmax(model(features), dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise ValueError("validation dataset must not be empty")
    return correct / total


def _split_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    validation_size = max(1, int(features.shape[0] * validation_fraction))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    return (
        TensorDataset(features[train_indices], labels[train_indices]),
        TensorDataset(features[validation_indices], labels[validation_indices]),
    )


def _save_checkpoint(model: ComplexityEstimatorNet, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 36,
            "output_dim": 4,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 complexity estimator checkpoint.")
    parser.add_argument("--samples-per-level", type=int, default=EstimatorTrainingConfig.samples_per_level)
    parser.add_argument("--epochs", type=int, default=EstimatorTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=EstimatorTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=EstimatorTrainingConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=EstimatorTrainingConfig.seed)
    parser.add_argument("--checkpoint-path", type=Path, default=EstimatorTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_estimator(
        EstimatorTrainingConfig(
            samples_per_level=args.samples_per_level,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            checkpoint_path=args.checkpoint_path,
        )
    )
    print(f"validation_accuracy={result.validation_accuracy:.4f}")
    print(f"checkpoint_path={result.checkpoint_path}")


if __name__ == "__main__":
    main()
