from __future__ import annotations

import torch

from models import IntentPredictorNet
from pipelines.train_intent_predictor import (
    IntentPredictorTrainingConfig,
    compute_intent_predictor_loss,
    generate_synthetic_intent_dataset,
    train_intent_predictor,
)


def test_synthetic_intent_dataset_contract() -> None:
    dataset = generate_synthetic_intent_dataset(samples=4, seed=31)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    assert batch[0].shape == (4, 256)
    assert batch[1].shape == (4,)
    assert batch[2].shape == (4,)
    assert batch[3].shape == (4, 2, 2)


def test_intent_predictor_loss_is_finite() -> None:
    dataset = generate_synthetic_intent_dataset(samples=4, seed=37)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_intent_predictor_loss(IntentPredictorNet(), batch)

    assert torch.isfinite(loss)


def test_train_intent_predictor_saves_checkpoint(tmp_path) -> None:
    result = train_intent_predictor(
        IntentPredictorTrainingConfig(
            samples=8,
            epochs=1,
            batch_size=4,
            checkpoint_path=tmp_path / "intent_predictor.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
