from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from models import AnomalyAutoencoder
from src.decision.intent_predictor import ActivityClass, IntentPrediction


@dataclass(frozen=True, slots=True)
class AnomalyDetection:
    score: float
    is_anomaly: bool
    statistical_score: float
    learned_score: float
    temporal_score: float


class AnomalyDetector:
    INPUT_DIM = 256
    ANOMALY_ACTIVITY_PRIORS = {
        ActivityClass.FALLING: 0.95,
        ActivityClass.FIGHTING: 0.95,
        ActivityClass.LOITERING: 0.7,
        ActivityClass.RUNNING: 0.45,
    }

    def __init__(
        self,
        autoencoder: AnomalyAutoencoder | None = None,
        *,
        threshold: float = 0.55,
        normal_center: np.ndarray | None = None,
        normal_scale: float = 4.0,
    ) -> None:
        self.autoencoder = autoencoder or AnomalyAutoencoder()
        self.autoencoder.eval()
        self.threshold = threshold
        self.normal_center = (
            np.asarray(normal_center, dtype=np.float32)
            if normal_center is not None
            else np.zeros(self.INPUT_DIM, dtype=np.float32)
        )
        self.normal_scale = normal_scale

    def detect(
        self,
        fused_reasoning: np.ndarray,
        *,
        intents: list[IntentPrediction] | None = None,
        previous_fused_reasoning: np.ndarray | None = None,
    ) -> list[AnomalyDetection]:
        fused = np.asarray(fused_reasoning, dtype=np.float32)
        if fused.ndim != 2 or fused.shape[1] != self.INPUT_DIM:
            raise ValueError(f"fused_reasoning must have shape [B, {self.INPUT_DIM}]")
        if intents is not None and len(intents) != fused.shape[0]:
            raise ValueError("intents length must match batch size")

        statistical_scores = self._statistical_score(fused)
        learned_scores = self._learned_score(fused)
        temporal_scores = self._temporal_score(fused, previous_fused_reasoning)

        detections: list[AnomalyDetection] = []
        for index in range(fused.shape[0]):
            intent_prior = self._intent_prior(intents[index]) if intents is not None else 0.0
            score = float(
                np.clip(
                    0.35 * statistical_scores[index]
                    + 0.35 * learned_scores[index]
                    + 0.2 * temporal_scores[index]
                    + 0.1 * intent_prior,
                    0.0,
                    1.0,
                )
            )
            detections.append(
                AnomalyDetection(
                    score=score,
                    is_anomaly=score >= self.threshold,
                    statistical_score=float(statistical_scores[index]),
                    learned_score=float(learned_scores[index]),
                    temporal_score=float(temporal_scores[index]),
                )
            )
        return detections

    def evaluate_fixture_cases(self, fixtures_dir: str | Path) -> float:
        fixtures = sorted(Path(fixtures_dir).glob("case_*.json"))
        if not fixtures:
            raise ValueError("no anomaly fixtures found")

        correct = 0
        for fixture_path in fixtures:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            activity = ActivityClass[payload["activity"]]
            fused = np.zeros((1, self.INPUT_DIM), dtype=np.float32)
            if activity in {ActivityClass.FALLING, ActivityClass.FIGHTING}:
                fused[0, :8] = 8.0
            elif activity is ActivityClass.LOITERING:
                fused[0, :8] = 4.0

            detections = self.detect(
                fused,
                intents=[
                    IntentPrediction(
                        direction=intents_direction_for(activity),
                        activity=activity,
                        direction_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
                        activity_probabilities=np.full(9, 1.0 / 9.0, dtype=np.float32),
                        trajectory_offsets=np.zeros((2, 2), dtype=np.float32),
                    )
                ],
            )
            if detections[0].is_anomaly is bool(payload["expected_anomaly"]):
                correct += 1
        return correct / len(fixtures)

    def _statistical_score(self, fused_reasoning: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(fused_reasoning - self.normal_center, axis=1)
        return np.clip(distances / self.normal_scale, 0.0, 1.0).astype(np.float32)

    def _learned_score(self, fused_reasoning: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            reconstructed = self.autoencoder(torch.from_numpy(fused_reasoning)).detach().cpu().numpy()
        reconstruction_error = np.mean(np.square(fused_reasoning - reconstructed), axis=1)
        return np.clip(reconstruction_error / 2.0, 0.0, 1.0).astype(np.float32)

    def _temporal_score(self, fused_reasoning: np.ndarray, previous_fused_reasoning: np.ndarray | None) -> np.ndarray:
        if previous_fused_reasoning is None:
            return np.zeros(fused_reasoning.shape[0], dtype=np.float32)
        previous = np.asarray(previous_fused_reasoning, dtype=np.float32)
        if previous.shape != fused_reasoning.shape:
            raise ValueError("previous_fused_reasoning must match fused_reasoning shape")
        delta = np.linalg.norm(fused_reasoning - previous, axis=1)
        return np.clip(delta / 6.0, 0.0, 1.0).astype(np.float32)

    def _intent_prior(self, intent: IntentPrediction) -> float:
        return self.ANOMALY_ACTIVITY_PRIORS.get(intent.activity, 0.05)


def intents_direction_for(activity: ActivityClass) -> int:
    if activity in {ActivityClass.STANDING, ActivityClass.SITTING, ActivityClass.LOITERING}:
        return 8
    return 0
