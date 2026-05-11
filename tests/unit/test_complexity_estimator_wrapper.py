from __future__ import annotations

import numpy as np
import pytest

from src.complexity.estimator import (
    ComplexityEstimator,
    ComplexityLevel,
    SceneComplexityMetrics,
)
from src.perception.sensor_fusion import FusedEntity


class ThresholdRuntime:
    def run(self, input_batch: np.ndarray) -> np.ndarray:
        logits = np.full((input_batch.shape[0], 4), -4.0, dtype=np.float32)
        scores = input_batch[:, 0] + input_batch[:, 1] + input_batch[:, 2] + (1.0 - input_batch[:, 3])
        levels = np.clip((scores * 1.25).astype(np.int64), 0, 3)
        logits[np.arange(input_batch.shape[0]), levels] = 4.0
        return logits


def _entity(track_id: int, velocity_xy: tuple[float, float], x: float = 0.0) -> FusedEntity:
    return FusedEntity(
        track_id=track_id,
        bbox_xywh=np.array([100.0, 80.0, 40.0, 120.0], dtype=np.float32),
        position_3d=np.array([x, 0.0, 2.0], dtype=np.float32),
        velocity_3d=np.array([velocity_xy[0], velocity_xy[1], 0.0], dtype=np.float32),
        heading_rad=0.0,
        confidence=0.9,
        nearest_obstacle_distance_m=1.5,
        nearest_obstacle_centroid_xy=np.array([x + 0.2, 0.0], dtype=np.float32),
        ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
    )


def test_complexity_estimator_builds_36_dim_input_vector() -> None:
    metrics = SceneComplexityMetrics(
        crowd_density=0.2,
        motion_entropy=0.3,
        anomaly_score_prev=0.4,
        soh_budget=0.9,
        scene_embedding=np.arange(32, dtype=np.float32),
    )

    vector = ComplexityEstimator.build_input_vector(metrics)

    assert vector.shape == (1, 36)
    assert vector.dtype == np.float32
    assert vector[0, :4].tolist() == pytest.approx([0.2, 0.3, 0.4, 0.9])
    assert np.array_equal(vector[0, 4:], np.arange(32, dtype=np.float32))


def test_complexity_estimator_extracts_scene_metrics_from_entities() -> None:
    entities = [
        _entity(1, (1.0, 0.0), x=0.0),
        _entity(2, (0.0, 1.0), x=1.0),
        _entity(3, (-1.0, 0.0), x=2.0),
    ]

    metrics = ComplexityEstimator.extract_scene_metrics(entities, soh_budget=1.2, max_entities=6)

    assert metrics.crowd_density == pytest.approx(0.5)
    assert metrics.motion_entropy > 0.0
    assert metrics.soh_budget == 1.0
    assert metrics.scene_embedding.shape == (32,)
    assert metrics.scene_embedding.dtype == np.float32


def test_complexity_estimator_predicts_synthetic_complexity_levels() -> None:
    estimator = ComplexityEstimator(runtime=ThresholdRuntime())
    low = estimator.estimate_from_metrics(
        SceneComplexityMetrics(
            crowd_density=0.05,
            motion_entropy=0.05,
            anomaly_score_prev=0.0,
            soh_budget=1.0,
            scene_embedding=np.zeros(32, dtype=np.float32),
        )
    )
    critical = estimator.estimate_from_metrics(
        SceneComplexityMetrics(
            crowd_density=0.9,
            motion_entropy=0.8,
            anomaly_score_prev=0.8,
            soh_budget=0.2,
            scene_embedding=np.ones(32, dtype=np.float32),
        )
    )

    assert low.level == ComplexityLevel.LOW
    assert critical.level == ComplexityLevel.CRITICAL
    assert low.probabilities.sum() == pytest.approx(1.0)
    assert critical.input_vector.shape == (36,)


def test_complexity_estimator_rejects_invalid_contracts() -> None:
    estimator = ComplexityEstimator(runtime=ThresholdRuntime())

    with pytest.raises(ValueError):
        ComplexityEstimator.build_input_vector(
            SceneComplexityMetrics(
                crowd_density=0.0,
                motion_entropy=0.0,
                anomaly_score_prev=0.0,
                soh_budget=1.0,
                scene_embedding=np.zeros(31, dtype=np.float32),
            )
        )
    with pytest.raises(ValueError):
        ComplexityEstimator.extract_scene_metrics([], soh_budget=1.0, max_entities=0)

    class BadRuntime:
        def run(self, input_batch: np.ndarray) -> np.ndarray:
            return np.zeros((input_batch.shape[0], 3), dtype=np.float32)

    with pytest.raises(ValueError):
        ComplexityEstimator(runtime=BadRuntime()).estimate_from_metrics(
            SceneComplexityMetrics(
                crowd_density=0.0,
                motion_entropy=0.0,
                anomaly_score_prev=0.0,
                soh_budget=1.0,
                scene_embedding=np.zeros(32, dtype=np.float32),
            )
        )

    assert estimator.estimate_from_entities([], soh_budget=1.0).level == ComplexityLevel.LOW
