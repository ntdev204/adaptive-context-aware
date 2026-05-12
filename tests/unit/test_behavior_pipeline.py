from __future__ import annotations

import numpy as np

from src.complexity.estimator import ComplexityEstimate, ComplexityLevel
from src.decision.behavior_pipeline import BehaviorDecisionPipeline
from src.decision.nav_commander import NavigationMode, RobotGoal
from src.perception.sensor_fusion import FusedEntity
from src.reasoning.brain_pipeline import BrainPipelineResult
from src.router.adaptive_router import ReasoningPathway, RoutingDecision


def _entity(track_id: int, position_xy: tuple[float, float]) -> FusedEntity:
    return FusedEntity(
        track_id=track_id,
        bbox_xywh=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        position_3d=np.array([position_xy[0], position_xy[1], 0.0], dtype=np.float32),
        velocity_3d=np.zeros(3, dtype=np.float32),
        heading_rad=0.0,
        confidence=0.95,
        nearest_obstacle_distance_m=2.0,
        nearest_obstacle_centroid_xy=np.array([2.0, 0.0], dtype=np.float32),
        ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
    )


def _brain_result(unified_output: np.ndarray) -> BrainPipelineResult:
    probabilities = np.zeros(4, dtype=np.float32)
    probabilities[2] = 1.0
    return BrainPipelineResult(
        estimate=ComplexityEstimate(
            level=ComplexityLevel.HIGH,
            logits=probabilities.copy(),
            probabilities=probabilities,
            input_vector=np.zeros(36, dtype=np.float32),
        ),
        routing=RoutingDecision(
            requested_level=ComplexityLevel.HIGH,
            effective_level=ComplexityLevel.HIGH,
            active_pathways=(ReasoningPathway.GRU, ReasoningPathway.TCN, ReasoningPathway.ATTENTION),
            latency_budget_ms=20.0,
            pathway_budget_ms={
                ReasoningPathway.GRU: 2.0,
                ReasoningPathway.TCN: 3.0,
                ReasoningPathway.ATTENTION: 8.0,
            },
            soh_budget=1.0,
        ),
        pathway_outputs={
            ReasoningPathway.GRU: np.zeros((1, 64), dtype=np.float32),
            ReasoningPathway.TCN: np.zeros((1, 64), dtype=np.float32),
            ReasoningPathway.ATTENTION: np.zeros((1, 128), dtype=np.float32),
        },
        unified_output=unified_output,
        latency_ms=5.0,
    )


def test_behavior_pipeline_produces_decision_outputs() -> None:
    pipeline = BehaviorDecisionPipeline()
    entities = [_entity(1, (0.8, 0.0)), _entity(2, (1.8, 0.5))]
    unified_output = np.zeros((1, 256), dtype=np.float32)
    unified_output[0, :8] = 6.0

    result = pipeline.run(
        _brain_result(unified_output),
        entities=entities,
        robot_goal=RobotGoal(target_xy_m=np.array([2.0, 0.0], dtype=np.float32), preferred_speed_mps=0.8),
        previous_unified_output=np.zeros((1, 256), dtype=np.float32),
    )

    assert len(result.intents) == 2
    assert len(result.anomaly_detections) == 2
    assert result.navigation_command.mode in {NavigationMode.AVOID, NavigationMode.HOLD}
    assert result.anomaly_detections[0].score >= 0.0


def test_behavior_pipeline_requires_valid_unified_output() -> None:
    pipeline = BehaviorDecisionPipeline()

    try:
        pipeline.run(
            _brain_result(np.zeros((2, 128), dtype=np.float32)),
            entities=[_entity(1, (1.0, 0.0))],
            robot_goal=RobotGoal(target_xy_m=np.array([1.0, 0.0], dtype=np.float32)),
        )
    except ValueError as exc:
        assert "unified_output" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid unified_output shape")
