from __future__ import annotations

import torch

from pipelines.train_fusion import (
    FusionBehaviorHead,
    FusionTrainingConfig,
    compute_fusion_loss,
    generate_synthetic_fusion_dataset,
    train_fusion,
)


def test_synthetic_fusion_dataset_contract() -> None:
    dataset = generate_synthetic_fusion_dataset(samples=4, seed=23)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    assert batch[0].shape == (4, 64)
    assert batch[1].shape == (4, 64)
    assert batch[2].shape == (4, 128)
    assert batch[3].shape == (4, 256)
    assert batch[4].shape == (4,)
    assert batch[5].shape == (4,)


def test_fusion_loss_is_finite() -> None:
    dataset = generate_synthetic_fusion_dataset(samples=4, seed=29)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_fusion_loss(FusionBehaviorHead(), batch)

    assert torch.isfinite(loss)


def test_train_fusion_saves_checkpoint(tmp_path) -> None:
    result = train_fusion(
        FusionTrainingConfig(
            samples=8,
            epochs=1,
            batch_size=4,
            checkpoint_path=tmp_path / "gated_fusion.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
