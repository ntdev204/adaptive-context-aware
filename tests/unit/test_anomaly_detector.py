from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.decision.anomaly_detector import AnomalyDetector
from src.decision.intent_predictor import ActivityClass, IntentDirection, IntentPrediction


def _intent(activity: ActivityClass) -> IntentPrediction:
    return IntentPrediction(
        direction=IntentDirection.NORTH,
        activity=activity,
        direction_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
        activity_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
        trajectory_offsets=np.zeros((2, 2), dtype=np.float32),
    )


def test_anomaly_detector_flags_strong_outlier() -> None:
    detector = AnomalyDetector()
    fused = np.zeros((1, 256), dtype=np.float32)
    fused[0, :8] = 9.0

    detections = detector.detect(fused, intents=[_intent(ActivityClass.FALLING)])

    assert detections[0].score >= detector.threshold
    assert detections[0].is_anomaly


def test_anomaly_detector_keeps_normal_case_below_threshold() -> None:
    detector = AnomalyDetector()
    fused = np.zeros((1, 256), dtype=np.float32)

    detections = detector.detect(fused, intents=[_intent(ActivityClass.WALKING)])

    assert detections[0].score < detector.threshold
    assert not detections[0].is_anomaly


def test_anomaly_detector_uses_temporal_change() -> None:
    detector = AnomalyDetector()
    current = np.ones((1, 256), dtype=np.float32) * 5.0
    previous = np.zeros((1, 256), dtype=np.float32)

    detections = detector.detect(current, previous_fused_reasoning=previous, intents=[_intent(ActivityClass.WALKING)])

    assert detections[0].temporal_score > 0.0


def test_anomaly_detector_fixture_recall_meets_target() -> None:
    detector = AnomalyDetector()
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "anomaly_synthetic"

    recall_like_accuracy = detector.evaluate_fixture_cases(fixtures_dir)

    assert recall_like_accuracy >= 0.8


def test_anomaly_detector_rejects_wrong_shapes() -> None:
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.detect(np.zeros((1, 128), dtype=np.float32))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((2, 256), dtype=np.float32), intents=[_intent(ActivityClass.WALKING)])
    with pytest.raises(ValueError):
        detector.detect(
            np.zeros((1, 256), dtype=np.float32),
            previous_fused_reasoning=np.zeros((2, 256), dtype=np.float32),
        )
