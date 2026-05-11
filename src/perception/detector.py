from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.runtime.tensorrt_engine import TensorRTEngineRunner

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


class TensorRTInferenceUnavailableError(RuntimeError):
    pass


class DetectorRuntime(Protocol):
    def run(self, input_batch: np.ndarray) -> np.ndarray:
        """Return raw detector output for one preprocessed batch."""


class PersonDetector:
    """Engine-only detector contract for Jetson runtime."""

    def __init__(self, config: DetectorConfig | None = None, runtime: DetectorRuntime | None = None) -> None:
        self.config = config or DetectorConfig()
        self._runtime = runtime

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape != (480, 640, 3):
            raise ValueError("expected BGR frame with shape (480, 640, 3)")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("expected uint8 BGR frame")

        chw = np.transpose(frame_bgr, (2, 0, 1)).astype(np.float32) / 255.0
        return chw[np.newaxis, ...]

    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> DetectorResult:
        input_batch = self.preprocess(frame_bgr)
        if self.config.backend == "synthetic":
            return DetectorResult(
                detections=self._infer_synthetic(frame_id),
                backend=self.config.backend,
            )
        try:
            raw_output = self._get_runtime().run(input_batch)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise TensorRTInferenceUnavailableError(f"TensorRT detector inference failed: {exc}") from exc
        detections = self._postprocess(raw_output, frame_shape=frame_bgr.shape)
        return DetectorResult(detections=self._filter_person_class(detections), backend=self.config.backend)

    def _engine_path(self) -> Path:
        if self.config.engine_path is not None:
            return self.config.engine_path
        return Path(os.environ.get("CTX_ENGINE_MODEL_PATH", "/app/models/engines/yolov8s.engine"))

    def _get_runtime(self) -> DetectorRuntime:
        if self._runtime is None:
            engine_path = self._engine_path()
            if not engine_path.exists():
                raise FileNotFoundError(f"missing TensorRT engine: {engine_path}")
            self._runtime = TensorRTEngineRunner(engine_path, ("images",))
        return self._runtime

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

    def _postprocess(self, raw_output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        output = np.asarray(raw_output, dtype=np.float32)
        if output.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise ValueError("detector runtime output must be rank 2 or batched rank 3")

        if output.shape[1] == 6:
            return self._postprocess_nms_output(output, frame_shape)
        if output.shape[0] >= 6 and output.shape[0] > output.shape[1]:
            output = output.T
        if output.shape[1] < 5:
            raise ValueError("detector runtime output must include boxes and class scores")
        return self._postprocess_yolov8_raw(output, frame_shape)

    def _postprocess_nms_output(self, output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        confidence_mask = output[:, 4] >= self.config.confidence_threshold
        detections = output[confidence_mask].copy()
        if detections.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy_mask = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
        detections[xyxy_mask, 2] -= detections[xyxy_mask, 0]
        detections[xyxy_mask, 3] -= detections[xyxy_mask, 1]
        return self._clip_detections(detections, frame_shape)

    def _postprocess_yolov8_raw(self, output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        boxes_xywh = output[:, :4]
        class_scores = output[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.float32)
        confidences = np.max(class_scores, axis=1)
        keep = (class_ids == PERSON_CLASS_ID) & (confidences >= self.config.confidence_threshold)
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes_xywh = boxes_xywh[keep].copy()
        boxes_xywh[:, 0] -= boxes_xywh[:, 2] / 2.0
        boxes_xywh[:, 1] -= boxes_xywh[:, 3] / 2.0
        detections = np.column_stack(
            [
                boxes_xywh,
                confidences[keep].astype(np.float32),
                class_ids[keep].astype(np.float32),
            ]
        ).astype(np.float32)
        detections = self._clip_detections(detections, frame_shape)
        keep_indices = _nms_xywh(detections[:, :4], detections[:, 4], iou_threshold=0.45)
        return detections[keep_indices].astype(np.float32, copy=False)

    @staticmethod
    def _clip_detections(detections: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        height, width, _ = frame_shape
        detections[:, 0] = np.clip(detections[:, 0], 0.0, float(width))
        detections[:, 1] = np.clip(detections[:, 1], 0.0, float(height))
        detections[:, 2] = np.clip(detections[:, 2], 0.0, float(width) - detections[:, 0])
        detections[:, 3] = np.clip(detections[:, 3], 0.0, float(height) - detections[:, 1])
        return detections.astype(np.float32, copy=False)


def _nms_xywh(boxes_xywh: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes_xywh.size == 0:
        return np.empty(0, dtype=np.int64)

    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    areas = np.maximum(0.0, boxes_xywh[:, 2]) * np.maximum(0.0, boxes_xywh[:, 3])
    order = np.argsort(scores)[::-1]

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersections = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        unions = areas[current] + areas[rest] - intersections
        iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)
