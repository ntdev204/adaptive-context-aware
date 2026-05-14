from __future__ import annotations

import torch

from pipelines.train_router_rl import RLTrainingConfig, generate_router_training_dataset, train_router_policy


def test_router_training_dataset_contract() -> None:
    features, labels = generate_router_training_dataset()

    assert features.shape == (20, 39)
    assert features.dtype == torch.float32
    assert labels.shape == (20,)
    assert labels.dtype == torch.long
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_train_router_policy_improves_fixture_reward_and_saves_checkpoint(tmp_path) -> None:
    result = train_router_policy(
        RLTrainingConfig(
            epochs=80,
            batch_size=8,
            learning_rate=0.02,
            seed=23,
            checkpoint_path=tmp_path / "rl_policy.pt",
            backend="supervised",
        )
    )

    assert result.validation_accuracy >= 0.75
    assert result.reward_after >= result.reward_before
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
