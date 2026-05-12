from __future__ import annotations

import torch

from pipelines.train_attention import (
    AttentionBehaviorHead,
    AttentionTrainingConfig,
    compute_attention_loss,
    generate_synthetic_attention_dataset,
    train_attention,
)


def test_synthetic_attention_dataset_contract() -> None:
    dataset = generate_synthetic_attention_dataset(samples=4, entity_count=5, seed=17)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    assert batch[0].shape == (4, 5, 128)
    assert batch[1].shape == (4,)
    assert batch[2].shape == (4,)


def test_attention_loss_is_finite() -> None:
    dataset = generate_synthetic_attention_dataset(samples=4, entity_count=5, seed=19)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_attention_loss(AttentionBehaviorHead(), batch)

    assert torch.isfinite(loss)


def test_train_attention_saves_checkpoint(tmp_path) -> None:
    result = train_attention(
        AttentionTrainingConfig(
            samples=8,
            entity_count=4,
            epochs=1,
            batch_size=4,
            checkpoint_path=tmp_path / "attention_pathway.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
