from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.sensor_fusion import FusedEntity
from src.reasoning.brain_pipeline import BrainPipelineResult
from src.utils.constants import UNIFIED_REASONING_DIM

from .anomaly_detector import AnomalyDetection, AnomalyDetector
from .intent_predictor import IntentPrediction, IntentPredictor
from .nav_commander import NavigationCommand, NavigationCommander, RobotGoal


@dataclass(frozen=True, slots=True)
class BehaviorDecisionResult:
    intents: list[IntentPrediction]
    anomaly_detections: list[AnomalyDetection]
    navigation_command: NavigationCommand


class BehaviorDecisionPipeline:
    def __init__(
        self,
        *,
        intent_predictor: IntentPredictor | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        navigation_commander: NavigationCommander | None = None,
    ) -> None:
        self.intent_predictor = intent_predictor or IntentPredictor()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.navigation_commander = navigation_commander or NavigationCommander()

    def run(
        self,
        brain_result: BrainPipelineResult,
        *,
        entities: list[FusedEntity],
        robot_goal: RobotGoal,
        previous_unified_output: np.ndarray | None = None,
    ) -> BehaviorDecisionResult:
        repeated_output = self._expand_unified_output(brain_result.unified_output, len(entities))
        intents = self.intent_predictor.predict(repeated_output)
        repeated_previous = (
            self._expand_unified_output(previous_unified_output, len(entities))
            if previous_unified_output is not None
            else None
        )
        anomaly_detections = self.anomaly_detector.detect(
            repeated_output,
            intents=intents,
            previous_fused_reasoning=repeated_previous,
        )
        navigation_command = self.navigation_commander.compute_command(
            entities=entities,
            intents=intents,
            anomaly_detections=anomaly_detections,
            robot_goal=robot_goal,
        )
        return BehaviorDecisionResult(
            intents=intents,
            anomaly_detections=anomaly_detections,
            navigation_command=navigation_command,
        )

    @staticmethod
    def _expand_unified_output(unified_output: np.ndarray | None, entity_count: int) -> np.ndarray:
        if unified_output is None:
            raise ValueError("unified_output is required")
        fused = np.asarray(unified_output, dtype=np.float32)
        if fused.ndim != 2 or fused.shape[1] != UNIFIED_REASONING_DIM:
            raise ValueError("unified_output must have shape [B, 256]")
        if entity_count <= 0:
            return fused
        if fused.shape[0] == entity_count:
            return fused
        if fused.shape[0] != 1:
            raise ValueError("unified_output batch must be 1 or match entity count")
        return np.repeat(fused, entity_count, axis=0)
