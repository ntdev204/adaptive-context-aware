from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from models import IntentPredictorNet


class IntentDirection(IntEnum):
    NORTH = 0
    NE = 1
    EAST = 2
    SE = 3
    SOUTH = 4
    SW = 5
    WEST = 6
    NW = 7
    STATIONARY = 8


class ActivityClass(IntEnum):
    WALKING = 0
    RUNNING = 1
    STANDING = 2
    SITTING = 3
    INTERACTING = 4
    FALLING = 5
    LOITERING = 6
    FIGHTING = 7
    OTHER = 8


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    direction: IntentDirection
    activity: ActivityClass
    direction_probabilities: np.ndarray
    activity_probabilities: np.ndarray
    trajectory_offsets: np.ndarray


class IntentPredictor:
    INPUT_DIM = 256

    def __init__(self, model: IntentPredictorNet | None = None) -> None:
        self.model = model or IntentPredictorNet()
        self.model.eval()

    def predict(self, fused_reasoning: np.ndarray) -> list[IntentPrediction]:
        fused = np.asarray(fused_reasoning, dtype=np.float32)
        if fused.ndim != 2 or fused.shape[1] != self.INPUT_DIM:
            raise ValueError(f"fused_reasoning must have shape [B, {self.INPUT_DIM}]")

        with torch.no_grad():
            outputs = self.model(torch.from_numpy(fused))

        direction_logits = outputs["direction_logits"].detach().cpu().numpy().astype(np.float32, copy=False)
        activity_logits = outputs["activity_logits"].detach().cpu().numpy().astype(np.float32, copy=False)
        trajectory_offsets = outputs["trajectory_offsets"].detach().cpu().numpy().astype(np.float32, copy=False)

        direction_probabilities = _softmax(direction_logits)
        activity_probabilities = _softmax(activity_logits)
        predictions: list[IntentPrediction] = []
        for index in range(fused.shape[0]):
            predictions.append(
                IntentPrediction(
                    direction=IntentDirection(int(np.argmax(direction_probabilities[index]))),
                    activity=ActivityClass(int(np.argmax(activity_probabilities[index]))),
                    direction_probabilities=direction_probabilities[index].copy(),
                    activity_probabilities=activity_probabilities[index].copy(),
                    trajectory_offsets=trajectory_offsets[index].copy(),
                )
            )
        return predictions


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)
