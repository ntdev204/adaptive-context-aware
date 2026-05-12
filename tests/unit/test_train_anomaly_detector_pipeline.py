from __future__ import annotations

import torch

from models import AnomalyAutoencoder
from pipelines.train_anomaly_detector import (
    AnomalyDetectorTrainingConfig,
    compute_anomaly_detector_loss,
    generate_synthetic_anomaly_dataset,
    train_anomaly_detector,
)


def test_synthetic_anomaly_dataset_contract() -> None:
    dataset = generate_synthetic_anomaly_dataset(samples=4, seed=41)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    assert batch[0].shape == (4, 256)
    assert batch[1].shape == (4,)


def test_anomaly_detector_loss_is_finite() -> None:
    dataset = generate_synthetic_anomaly_dataset(samples=4, seed=43)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_anomaly_detector_loss(AnomalyAutoencoder(), batch)

    assert torch.isfinite(loss)


def test_train_anomaly_detector_saves_checkpoint(tmp_path) -> None:
    result = train_anomaly_detector(
        AnomalyDetectorTrainingConfig(
            samples=8,
            epochs=1,
            batch_size=4,
            checkpoint_path=tmp_path / "anomaly_detector.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
