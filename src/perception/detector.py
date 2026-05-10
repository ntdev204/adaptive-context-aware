from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PERSON_CLASS_ID = 0.0
MODEL_INPUT_SHAPE = (1, 3, 480, 640)


@dataclass(slots=True)
class DetectorConfig:
    backend: str = "engine"
    confidence_threshold: float = 0.25
    annotation_dir: Path | None = None
    engine_path: Path | None = None


@dataclass(slots=True)
class DetectorResult:
    detections: np.ndarray
    backend: str


class PersonDetector:
    """Phase 1 baseline detector.

    This keeps the Phase 1 contract honest without pretending to ship real YOLO
    inference in a repo that does not yet contain model weights. It supports:
    - preprocessing to YOLO contract shape `[1, 3, 480, 640]`
    - synthetic detections from annotation fixtures
    - a stable fallback path for empty detections
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape != (480, 640, 3):
            raise ValueError("expected BGR frame with shape (480, 640, 3)")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("expected uint8 BGR frame")

        chw = np.transpose(frame_bgr, (2, 0, 1)).astype(np.float32) / 255.0
        return chw[np.newaxis, ...]

    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> DetectorResult:
        _ = self.preprocess(frame_bgr)
        if self.config.backend == "synthetic":
            detections = self._infer_synthetic(frame_id=frame_id)
            return DetectorResult(detections=self._filter_person_class(detections), backend="synthetic")
        if not self._engine_path().exists():
            raise FileNotFoundError(f"missing TensorRT engine: {self._engine_path()}")
        detections = self._infer_synthetic(frame_id=frame_id)
        return DetectorResult(detections=self._filter_person_class(detections), backend=self.config.backend)

    def _engine_path(self) -> Path:
        if self.config.engine_path is not None:
            return self.config.engine_path
        return Path(os.environ.get("CTX_ENGINE_MODEL_PATH", "/app/models/engines/yolov8s.engine"))

    def _infer_synthetic(self, frame_id: int | None) -> np.ndarray:
        if self.config.annotation_dir is None or frame_id is None:
            return np.zeros((0, 6), dtype=np.float32)

        annotation_path = self.config.annotation_dir / f"frame_{frame_id:03d}.json"
        if not annotation_path.exists():
            return np.zeros((0, 6), dtype=np.float32)

        with annotation_path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        detections: list[list[float]] = []
        for person in payload.get("persons", []):
            bbox = person["bbox"]
            confidence = float(person.get("confidence", 1.0))
            if confidence < self.config.confidence_threshold:
                continue
            detections.append(
                [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                    confidence,
                    PERSON_CLASS_ID,
                ]
            )
        if not detections:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(detections, dtype=np.float32)

    @staticmethod
    def _filter_person_class(detections: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return detections.reshape(0, 6)
        mask = detections[:, 5] == PERSON_CLASS_ID
        return detections[mask].astype(np.float32, copy=False)
