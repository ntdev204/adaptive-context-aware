from __future__ import annotations

import torch

from pipelines.train_tcn import (
    TcnBehaviorHead,
    TcnTrainingConfig,
    compute_tcn_loss,
    generate_synthetic_tcn_dataset,
    train_tcn,
)


def test_synthetic_tcn_dataset_contract() -> None:
    dataset = generate_synthetic_tcn_dataset(samples=4, sequence_length=8, seed=11)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    assert batch[0].shape == (4, 128, 8)
    assert batch[1].shape == (4,)
    assert batch[2].shape == (4,)


def test_tcn_loss_is_finite() -> None:
    dataset = generate_synthetic_tcn_dataset(samples=4, sequence_length=8, seed=13)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_tcn_loss(TcnBehaviorHead(), batch)

    assert torch.isfinite(loss)


def test_train_tcn_saves_checkpoint(tmp_path) -> None:
    result = train_tcn(
        TcnTrainingConfig(
            samples=8,
            sequence_length=8,
            epochs=1,
            batch_size=4,
            checkpoint_path=tmp_path / "tcn_pathway.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
