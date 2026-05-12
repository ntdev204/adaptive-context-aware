from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol

import numpy as np

from src.perception.sensor_fusion import FusedEntity
from src.runtime.tensorrt_engine import TensorRTEngineRunner


class ComplexityLevel(IntEnum):
    LOW = 0
    MED = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class SceneComplexityMetrics:
    crowd_density: float
    motion_entropy: float
    anomaly_score_prev: float
    soh_budget: float
    scene_embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class ComplexityEstimate:
    level: ComplexityLevel
    logits: np.ndarray
    probabilities: np.ndarray
    input_vector: np.ndarray


class EstimatorRuntime(Protocol):
    def run(self, input_batch: np.ndarray) -> np.ndarray:
        """Return logits for a `[B, 36]` float32 input batch."""


class ComplexityEstimator:
    DEFAULT_MODEL_PATH = Path("models/engines/estimator.engine")
    INPUT_DIM = 36
    SCENE_EMBEDDING_DIM = 32

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime: EstimatorRuntime | None = None,
    ) -> None:
        self.runtime = runtime or _TensorRTEstimatorRuntime(Path(model_path))

    def estimate_from_entities(
        self,
        entities: Sequence[FusedEntity],
        *,
        soh_budget: float,
        anomaly_score_prev: float = 0.0,
        scene_embedding: np.ndarray | None = None,
        max_entities: int = 20,
    ) -> ComplexityEstimate:
        metrics = self.extract_scene_metrics(
            entities,
            soh_budget=soh_budget,
            anomaly_score_prev=anomaly_score_prev,
            scene_embedding=scene_embedding,
            max_entities=max_entities,
        )
        return self.estimate_from_metrics(metrics)

    def estimate_from_metrics(self, metrics: SceneComplexityMetrics) -> ComplexityEstimate:
        input_vector = self.build_input_vector(metrics)
        logits = np.asarray(self.runtime.run(input_vector), dtype=np.float32)
        if logits.shape != (input_vector.shape[0], len(ComplexityLevel)):
            raise ValueError("complexity runtime must return logits with shape [B, 4]")

        probabilities = _softmax(logits)
        level = ComplexityLevel(int(np.argmax(probabilities[0])))
        return ComplexityEstimate(
            level=level,
            logits=logits[0].copy(),
            probabilities=probabilities[0].copy(),
            input_vector=input_vector[0].copy(),
        )

    @classmethod
    def build_input_vector(cls, metrics: SceneComplexityMetrics) -> np.ndarray:
        scene_embedding = np.asarray(metrics.scene_embedding, dtype=np.float32)
        if scene_embedding.shape != (cls.SCENE_EMBEDDING_DIM,):
            raise ValueError("scene_embedding must have shape [32]")

        input_vector = np.concatenate(
            [
                np.array(
                    [
                        _clip01(metrics.crowd_density),
                        _clip01(metrics.motion_entropy),
                        _clip01(metrics.anomaly_score_prev),
                        _clip01(metrics.soh_budget),
                    ],
                    dtype=np.float32,
                ),
                scene_embedding,
            ]
        )
        return input_vector.reshape(1, cls.INPUT_DIM).astype(np.float32, copy=False)

    @classmethod
    def extract_scene_metrics(
        cls,
        entities: Sequence[FusedEntity],
        *,
        soh_budget: float,
        anomaly_score_prev: float = 0.0,
        scene_embedding: np.ndarray | None = None,
        max_entities: int = 20,
    ) -> SceneComplexityMetrics:
        if max_entities <= 0:
            raise ValueError("max_entities must be positive")

        crowd_density = min(len(entities) / max_entities, 1.0)
        embedding = (
            np.asarray(scene_embedding, dtype=np.float32)
            if scene_embedding is not None
            else cls._build_scene_embedding(entities, crowd_density)
        )
        return SceneComplexityMetrics(
            crowd_density=crowd_density,
            motion_entropy=_motion_entropy(entities),
            anomaly_score_prev=_clip01(anomaly_score_prev),
            soh_budget=_clip01(soh_budget),
            scene_embedding=embedding,
        )

    @classmethod
    def _build_scene_embedding(cls, entities: Sequence[FusedEntity], crowd_density: float) -> np.ndarray:
        if not entities:
            return np.zeros(cls.SCENE_EMBEDDING_DIM, dtype=np.float32)

        positions = np.stack([np.asarray(entity.position_3d, dtype=np.float32) for entity in entities])
        velocities = np.stack([np.asarray(entity.velocity_3d, dtype=np.float32) for entity in entities])
        speeds = np.linalg.norm(velocities, axis=1)
        obstacle_distances = np.array(
            [
                entity.nearest_obstacle_distance_m if entity.nearest_obstacle_distance_m is not None else 10.0
                for entity in entities
            ],
            dtype=np.float32,
        )
        confidences = np.array([entity.confidence for entity in entities], dtype=np.float32)
        extent_xy = np.ptp(positions[:, :2], axis=0)

        base = np.array(
            [
                crowd_density,
                float(np.mean(speeds)),
                float(np.max(speeds)),
                float(np.std(speeds)),
                float(np.mean(positions[:, 0])),
                float(np.mean(positions[:, 1])),
                float(np.mean(positions[:, 2])),
                float(np.std(positions[:, 0])),
                float(np.std(positions[:, 1])),
                float(np.std(positions[:, 2])),
                float(np.mean(confidences)),
                float(np.min(obstacle_distances)),
                float(np.mean(obstacle_distances < 2.0)),
                float(np.linalg.norm(extent_xy)),
                float(len(entities)),
                _motion_entropy(entities),
            ],
            dtype=np.float32,
        )
        return np.tile(base, 2)[: cls.SCENE_EMBEDDING_DIM].astype(np.float32, copy=False)


class _TensorRTEstimatorRuntime:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"complexity estimator TensorRT engine not found: {model_path}")
        self.model_path = model_path
        self.runner = TensorRTEngineRunner(model_path, ("complexity_features",))

    def run(self, input_batch: np.ndarray) -> np.ndarray:
        return self.runner.run(input_batch)


def _motion_entropy(entities: Sequence[FusedEntity]) -> float:
    if not entities:
        return 0.0

    velocities = np.stack([np.asarray(entity.velocity_3d[:2], dtype=np.float32) for entity in entities])
    speeds = np.linalg.norm(velocities, axis=1)
    moving = speeds > 1e-4
    if int(np.count_nonzero(moving)) <= 1:
        return 0.0

    angles = np.arctan2(velocities[moving, 1], velocities[moving, 0])
    bins = np.linspace(-np.pi, np.pi, num=9, dtype=np.float32)
    counts, _ = np.histogram(angles, bins=bins)
    probabilities = counts[counts > 0].astype(np.float32)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return _clip01(entropy / np.log(8.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
