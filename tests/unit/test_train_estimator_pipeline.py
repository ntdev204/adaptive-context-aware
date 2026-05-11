from __future__ import annotations

import torch

from pipelines.train_estimator import (
    EstimatorTrainingConfig,
    generate_synthetic_complexity_dataset,
    train_estimator,
)


def test_synthetic_estimator_dataset_contract() -> None:
    features, labels = generate_synthetic_complexity_dataset(samples_per_level=5, seed=11)

    assert features.shape == (20, 36)
    assert features.dtype == torch.float32
    assert labels.shape == (20,)
    assert labels.dtype == torch.long
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_train_estimator_reaches_validation_accuracy_and_saves_checkpoint(tmp_path) -> None:
    result = train_estimator(
        EstimatorTrainingConfig(
            samples_per_level=32,
            epochs=80,
            batch_size=16,
            learning_rate=0.02,
            seed=13,
            checkpoint_path=tmp_path / "complexity_estimator.pt",
            onnx_path=tmp_path / "estimator.onnx",
            export_onnx=False,
        )
    )

    assert result.validation_accuracy >= 0.90
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.onnx_path is None


def test_train_estimator_rejects_invalid_dataset_size() -> None:
    try:
        generate_synthetic_complexity_dataset(samples_per_level=0)
    except ValueError as exc:
        assert "samples_per_level" in str(exc)
    else:
        raise AssertionError("expected invalid samples_per_level to raise ValueError")
