from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.perception.sensor_fusion import FusedEntity

from .anomaly_detector import AnomalyDetection
from .intent_predictor import IntentDirection, IntentPrediction


class NavigationMode(StrEnum):
    PROCEED = "proceed"
    AVOID = "avoid"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class RobotGoal:
    target_xy_m: np.ndarray
    preferred_speed_mps: float = 0.8


@dataclass(frozen=True, slots=True)
class NavigationCommand:
    velocity_xy_mps: np.ndarray
    omega_radps: float
    mode: NavigationMode
    reason: str


class NavigationCommander:
    CRITICAL_ANOMALY_THRESHOLD = 0.85
    COLLISION_DISTANCE_THRESHOLD_M = 1.25
    CAUTION_ANOMALY_THRESHOLD = 0.55

    def compute_command(
        self,
        *,
        entities: list[FusedEntity],
        intents: list[IntentPrediction],
        anomaly_detections: list[AnomalyDetection],
        robot_goal: RobotGoal,
    ) -> NavigationCommand:
        if len(intents) != len(entities) or len(anomaly_detections) != len(entities):
            raise ValueError("entities, intents, and anomaly_detections must have the same length")

        goal = np.asarray(robot_goal.target_xy_m, dtype=np.float32)
        if goal.shape != (2,):
            raise ValueError("robot_goal.target_xy_m must have shape [2]")

        if any(detection.score >= self.CRITICAL_ANOMALY_THRESHOLD for detection in anomaly_detections):
            return NavigationCommand(
                velocity_xy_mps=np.zeros(2, dtype=np.float32),
                omega_radps=0.0,
                mode=NavigationMode.HOLD,
                reason="critical anomaly detected",
            )

        desired_velocity = _normalize(goal) * robot_goal.preferred_speed_mps
        avoidance = np.zeros(2, dtype=np.float32)
        highest_risk = 0.0
        highest_heading_bias = 0.0

        for entity, intent, anomaly in zip(entities, intents, anomaly_detections, strict=True):
            relative_xy = np.asarray(entity.position_3d[:2], dtype=np.float32)
            distance = float(np.linalg.norm(relative_xy))
            if distance <= 1e-6:
                continue

            predicted_velocity = _direction_unit_vector(intent.direction) * np.linalg.norm(entity.velocity_3d[:2])
            predicted_position = relative_xy + predicted_velocity + intent.trajectory_offsets[0]
            risk = self._collision_risk(predicted_position, distance=distance, anomaly_score=anomaly.score)
            if anomaly.is_anomaly and distance <= self.COLLISION_DISTANCE_THRESHOLD_M:
                risk = max(risk, 0.45)
            if risk <= 0.0:
                continue

            highest_risk = max(highest_risk, risk)
            away_vector = _normalize(-predicted_position)
            lateral_vector = np.array([-away_vector[1], away_vector[0]], dtype=np.float32)
            avoidance += away_vector * risk + lateral_vector * (0.15 * risk)
            highest_heading_bias += float(np.sign(predicted_position[1]) * risk)

        blended_velocity = desired_velocity + avoidance
        speed_limit = max(0.0, robot_goal.preferred_speed_mps)
        speed = float(np.linalg.norm(blended_velocity))
        if speed > speed_limit > 0.0:
            blended_velocity = blended_velocity / speed * speed_limit

        if highest_risk >= 0.45:
            return NavigationCommand(
                velocity_xy_mps=blended_velocity.astype(np.float32, copy=False),
                omega_radps=float(np.clip(-highest_heading_bias, -1.0, 1.0)),
                mode=NavigationMode.AVOID,
                reason="predicted collision path detected",
            )

        return NavigationCommand(
            velocity_xy_mps=desired_velocity.astype(np.float32, copy=False),
            omega_radps=0.0,
            mode=NavigationMode.PROCEED,
            reason="path is clear",
        )

    def _collision_risk(self, predicted_position: np.ndarray, *, distance: float, anomaly_score: float) -> float:
        predicted_distance = float(np.linalg.norm(predicted_position))
        if predicted_distance > self.COLLISION_DISTANCE_THRESHOLD_M and anomaly_score < 0.5:
            return 0.0

        proximity = np.clip(1.0 - predicted_distance / self.COLLISION_DISTANCE_THRESHOLD_M, 0.0, 1.0)
        distance_weight = np.clip(1.0 - distance / (self.COLLISION_DISTANCE_THRESHOLD_M * 2.0), 0.0, 1.0)
        return float(np.clip(0.65 * proximity + 0.2 * distance_weight + 0.15 * anomaly_score, 0.0, 1.0))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


_DIRECTION_UNIT_VECTORS: dict[IntentDirection, np.ndarray] = {
    IntentDirection.NORTH: np.array([0.0, 1.0], dtype=np.float32),
    IntentDirection.NE: _normalize(np.array([1.0, 1.0], dtype=np.float32)),
    IntentDirection.EAST: np.array([1.0, 0.0], dtype=np.float32),
    IntentDirection.SE: _normalize(np.array([1.0, -1.0], dtype=np.float32)),
    IntentDirection.SOUTH: np.array([0.0, -1.0], dtype=np.float32),
    IntentDirection.SW: _normalize(np.array([-1.0, -1.0], dtype=np.float32)),
    IntentDirection.WEST: np.array([-1.0, 0.0], dtype=np.float32),
    IntentDirection.NW: _normalize(np.array([-1.0, 1.0], dtype=np.float32)),
    IntentDirection.STATIONARY: np.zeros(2, dtype=np.float32),
}


def _direction_unit_vector(direction: IntentDirection) -> np.ndarray:
    return _DIRECTION_UNIT_VECTORS[direction]
