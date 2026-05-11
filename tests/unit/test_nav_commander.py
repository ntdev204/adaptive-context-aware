from __future__ import annotations

import numpy as np

from src.decision.anomaly_detector import AnomalyDetection
from src.decision.intent_predictor import ActivityClass, IntentDirection, IntentPrediction
from src.decision.nav_commander import NavigationCommander, NavigationMode, RobotGoal
from src.perception.sensor_fusion import FusedEntity


def _entity(track_id: int, position_xy: tuple[float, float], velocity_xy: tuple[float, float]) -> FusedEntity:
    return FusedEntity(
        track_id=track_id,
        bbox_xywh=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        position_3d=np.array([position_xy[0], position_xy[1], 0.0], dtype=np.float32),
        velocity_3d=np.array([velocity_xy[0], velocity_xy[1], 0.0], dtype=np.float32),
        heading_rad=0.0,
        confidence=0.9,
        nearest_obstacle_distance_m=2.0,
        nearest_obstacle_centroid_xy=np.array([2.0, 0.0], dtype=np.float32),
        ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
    )


def _intent(direction: IntentDirection, activity: ActivityClass = ActivityClass.WALKING) -> IntentPrediction:
    return IntentPrediction(
        direction=direction,
        activity=activity,
        direction_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
        activity_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
        trajectory_offsets=np.zeros((2, 2), dtype=np.float32),
    )


def _anomaly(score: float) -> AnomalyDetection:
    return AnomalyDetection(
        score=score,
        is_anomaly=score >= 0.55,
        statistical_score=score,
        learned_score=score,
        temporal_score=score,
    )


def test_nav_commander_proceeds_when_path_is_clear() -> None:
    commander = NavigationCommander()

    command = commander.compute_command(
        entities=[],
        intents=[],
        anomaly_detections=[],
        robot_goal=RobotGoal(target_xy_m=np.array([2.0, 0.0], dtype=np.float32), preferred_speed_mps=0.8),
    )

    assert command.mode == NavigationMode.PROCEED
    assert command.velocity_xy_mps[0] > 0.0
    assert command.reason == "path is clear"


def test_nav_commander_avoids_predicted_collision() -> None:
    commander = NavigationCommander()
    entity = _entity(1, (0.6, 0.0), (0.0, 0.0))

    command = commander.compute_command(
        entities=[entity],
        intents=[_intent(IntentDirection.STATIONARY)],
        anomaly_detections=[_anomaly(0.2)],
        robot_goal=RobotGoal(target_xy_m=np.array([2.0, 0.0], dtype=np.float32), preferred_speed_mps=0.8),
    )

    assert command.mode == NavigationMode.AVOID
    assert command.velocity_xy_mps[1] != 0.0 or command.velocity_xy_mps[0] < 0.8
    assert command.reason == "predicted collision path detected"


def test_nav_commander_holds_on_critical_anomaly() -> None:
    commander = NavigationCommander()
    entity = _entity(1, (1.0, 0.0), (0.0, 0.0))

    command = commander.compute_command(
        entities=[entity],
        intents=[_intent(IntentDirection.NORTH, ActivityClass.FALLING)],
        anomaly_detections=[_anomaly(0.95)],
        robot_goal=RobotGoal(target_xy_m=np.array([2.0, 0.0], dtype=np.float32), preferred_speed_mps=0.8),
    )

    assert command.mode == NavigationMode.HOLD
    assert np.allclose(command.velocity_xy_mps, np.zeros(2, dtype=np.float32))
    assert command.reason == "critical anomaly detected"
