from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import AttentionPathway


@dataclass(frozen=True, slots=True)
class AttentionTrainingConfig:
    samples: int = 96
    entity_count: int = 6
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 35
    checkpoint_path: Path = Path("models/checkpoints/attention_pathway.pt")


@dataclass(frozen=True, slots=True)
class AttentionTrainingResult:
    final_loss: float
    epochs: int
    checkpoint_path: Path


class AttentionBehaviorHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = AttentionPathway()
        self.interaction_head = nn.Linear(128, 4)
        self.risk_head = nn.Linear(128, 1)

    def forward(self, entity_features: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.attention(entity_features)
        return {
            "context": context,
            "interaction_logits": self.interaction_head(context),
            "risk": self.risk_head(context).squeeze(-1),
        }


def generate_synthetic_attention_dataset(
    samples: int = 96,
    entity_count: int = 6,
    seed: int = 35,
) -> TensorDataset:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if entity_count <= 0:
        raise ValueError("entity_count must be positive")

    generator = torch.Generator().manual_seed(seed)
    interaction_labels = torch.arange(samples, dtype=torch.long) % 4
    risk_targets = interaction_labels.float() / 3.0
    entity_features = torch.randn(samples, entity_count, 128, generator=generator) * 0.05
    entity_features[:, :, 0] = interaction_labels.float().unsqueeze(1) / 3.0
    entity_features[:, :, 1] = risk_targets.unsqueeze(1)
    return TensorDataset(entity_features, interaction_labels, risk_targets)


def compute_attention_loss(model: AttentionBehaviorHead, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    entity_features, interaction_labels, risk_targets = batch
    outputs = model(entity_features)
    interaction_loss = nn.functional.cross_entropy(outputs["interaction_logits"], interaction_labels)
    risk_loss = nn.functional.mse_loss(outputs["risk"], risk_targets)
    return interaction_loss + 0.2 * risk_loss


def train_attention(config: AttentionTrainingConfig = AttentionTrainingConfig()) -> AttentionTrainingResult:
    torch.manual_seed(config.seed)
    dataset = generate_synthetic_attention_dataset(
        samples=config.samples,
        entity_count=config.entity_count,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = AttentionBehaviorHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_attention_loss(model, batch)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

    checkpoint_path = _save_checkpoint(model, config.checkpoint_path)
    return AttentionTrainingResult(final_loss=final_loss, epochs=config.epochs, checkpoint_path=checkpoint_path)


def _save_checkpoint(model: AttentionBehaviorHead, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the attention pathway on synthetic interaction data.")
    parser.add_argument("--samples", type=int, default=AttentionTrainingConfig.samples)
    parser.add_argument("--entity-count", type=int, default=AttentionTrainingConfig.entity_count)
    parser.add_argument("--epochs", type=int, default=AttentionTrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=AttentionTrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=AttentionTrainingConfig.learning_rate)
    parser.add_argument("--checkpoint-path", type=Path, default=AttentionTrainingConfig.checkpoint_path)
    args = parser.parse_args()

    result = train_attention(
        AttentionTrainingConfig(
            samples=args.samples,
            entity_count=args.entity_count,
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
