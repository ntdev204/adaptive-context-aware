from __future__ import annotations

import torch

from pipelines.train_reasoning import (
    JointReasoningNet,
    ReasoningTrainingConfig,
    compute_reasoning_loss,
    generate_synthetic_reasoning_dataset,
    train_reasoning,
)


def test_joint_reasoning_model_output_contract() -> None:
    dataset = generate_synthetic_reasoning_dataset(samples=4, sequence_length=8, entity_count=5, seed=31)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    model = JointReasoningNet()

    outputs = model(batch[0], batch[1], batch[2])

    assert outputs["fused"].shape == (4, 256)
    assert outputs["activity_logits"].shape == (4, 4)
    assert outputs["direction_logits"].shape == (4, 9)
    assert outputs["anomaly_logits"].shape == (4,)
    assert outputs["reconstruction"].shape == (4, 128)


def test_reasoning_loss_is_finite() -> None:
    dataset = generate_synthetic_reasoning_dataset(samples=4, sequence_length=8, entity_count=5, seed=37)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))

    loss = compute_reasoning_loss(JointReasoningNet(), batch)

    assert torch.isfinite(loss)


def test_train_reasoning_saves_checkpoint(tmp_path) -> None:
    result = train_reasoning(
        ReasoningTrainingConfig(
            samples=8,
            sequence_length=8,
            entity_count=4,
            epochs=1,
            batch_size=4,
            seed=41,
            checkpoint_path=tmp_path / "reasoning_brain.pt",
        )
    )

    assert result.final_loss > 0.0
    assert result.checkpoint_path.exists()
    assert result.epochs == 1


def test_reasoning_dataset_rejects_invalid_sizes() -> None:
    try:
        generate_synthetic_reasoning_dataset(samples=0)
    except ValueError as exc:
        assert "samples" in str(exc)
    else:
        raise AssertionError("expected invalid samples to raise ValueError")
