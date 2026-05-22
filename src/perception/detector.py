from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.runtime.tensorrt_engine import TensorRTEngineRunner
from src.utils.constants import FRAME_HEIGHT, FRAME_WIDTH

PERSON_CLASS_ID = 0.0
MODEL_INPUT_SHAPE = (1, 3, FRAME_HEIGHT, FRAME_WIDTH)


def _empty_detections() -> np.ndarray:
    """Return a fresh empty detections array with shape ``[0, 6]``."""
    return np.zeros((0, 6), dtype=np.float32)


@dataclass(slots=True)
class DetectorConfig:
    backend: str = "engine"
    """One of: ``"engine"``, ``"pt"``, or ``"synthetic"``."""
    confidence_threshold: float = 0.25
    annotation_dir: Path | None = None
    engine_path: Path | None = None
    """Path to ``.engine`` file (engine backend). Falls back to ``CTX_ENGINE_MODEL_PATH`` env var."""
    pt_model_path: Path | None = None
    """Path to ``.pt`` detector weights (pt backend). Falls back to ``CTX_PT_MODEL_PATH`` env var."""


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
    """Multi-backend person detector.

    Supported backends (set via ``DetectorConfig.backend``):

    * ``"engine"``    – TensorRT ``.engine`` file via :class:`TensorRTEngineRunner`
                        (optimised for Jetson / CUDA devices).
    * ``"synthetic"`` – JSON annotation files used for offline unit tests.
    """

    def __init__(self, config: DetectorConfig | None = None, runtime: DetectorRuntime | None = None) -> None:
        self.config = config or DetectorConfig()
        self._runtime = runtime
        self._pt_model: Any | None = None

    def warmup(self) -> None:
        if self.config.backend != "engine":
            return
        dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.detect(dummy)

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if close is not None:
            close()
        self._runtime = None

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ValueError(f"expected BGR frame with shape ({FRAME_HEIGHT}, {FRAME_WIDTH}, 3)")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("expected uint8 BGR frame")

        chw = np.transpose(frame_bgr, (2, 0, 1)).astype(np.float32) / 255.0
        return chw[np.newaxis, ...]

    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> DetectorResult:
        # --- synthetic backend ---
        if self.config.backend == "synthetic":
            return DetectorResult(
                detections=self._infer_synthetic(frame_id),
                backend=self.config.backend,
            )

        if self.config.backend == "pt":
            detections = self._infer_pt(frame_bgr)
            return DetectorResult(detections=detections, backend=self.config.backend)

        # --- engine (TensorRT) backend ---
        input_batch = self.preprocess(frame_bgr)
        try:
            raw_output = self._get_runtime().run(input_batch)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise TensorRTInferenceUnavailableError(f"TensorRT detector inference failed: {exc}") from exc
        detections = self._postprocess(raw_output, frame_shape=frame_bgr.shape)
        return DetectorResult(detections=self._filter_person_class(detections), backend=self.config.backend)

    # ------------------------------------------------------------------
    # Engine (TensorRT) backend helpers
    # ------------------------------------------------------------------

    def _engine_path(self) -> Path:
        if self.config.engine_path is not None:
            return self.config.engine_path
        return Path(os.environ.get("CTX_ENGINE_MODEL_PATH", "/app/models/engines/best.engine"))

    def _get_runtime(self) -> DetectorRuntime:
        if self._runtime is None:
            engine_path = self._engine_path()
            if not engine_path.exists():
                raise FileNotFoundError(f"missing TensorRT engine: {engine_path}")
            self._runtime = TensorRTEngineRunner(engine_path, ("images",))
        return self._runtime

    def _pt_model_path(self) -> Path:
        if self.config.pt_model_path is not None:
            return self.config.pt_model_path
        return Path(os.environ.get("CTX_PT_MODEL_PATH", "/app/models/fine_tuning/best.pt"))

    def _get_pt_model(self) -> Any:
        if self._pt_model is None:
            model_path = self._pt_model_path()
            if not model_path.exists():
                raise FileNotFoundError(f"missing detector weights: {model_path}")
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for pt detector backend") from exc
            self._pt_model = YOLO(str(model_path))
        return self._pt_model

    def _infer_pt(self, frame_bgr: np.ndarray) -> np.ndarray:
        model = self._get_pt_model()
        results = model.predict(frame_bgr, conf=self.config.confidence_threshold, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return _empty_detections()

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.cpu().numpy().astype(np.float32)
        xywh = xyxy.copy()
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        detections = np.column_stack([xywh, confidences, class_ids]).astype(np.float32)
        return self._filter_person_class(self._clip_detections(detections, frame_bgr.shape))

    # ------------------------------------------------------------------
    # Synthetic backend helper
    # ------------------------------------------------------------------

    def _infer_synthetic(self, frame_id: int | None) -> np.ndarray:
        if self.config.annotation_dir is None or frame_id is None:
            return _empty_detections()

        annotation_path = self.config.annotation_dir / f"frame_{frame_id:03d}.json"
        if not annotation_path.exists():
            return _empty_detections()

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
            return _empty_detections()
        return np.asarray(detections, dtype=np.float32)

    # ------------------------------------------------------------------
    # Shared postprocessing (engine backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_person_class(detections: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return detections.reshape(0, 6)
        mask = detections[:, 5] == PERSON_CLASS_ID
        return detections[mask].astype(np.float32, copy=False)

    def _postprocess(self, raw_output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        output = np.asarray(raw_output, dtype=np.float32)
        if output.size == 0:
            return _empty_detections()

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
        return self._postprocess_yolo11_raw(output, frame_shape)

    def _postprocess_nms_output(self, output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        confidence_mask = output[:, 4] >= self.config.confidence_threshold
        detections = output[confidence_mask].copy()
        if detections.size == 0:
            return _empty_detections()

        xyxy_mask = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
        detections[xyxy_mask, 2] -= detections[xyxy_mask, 0]
        detections[xyxy_mask, 3] -= detections[xyxy_mask, 1]
        return self._clip_detections(detections, frame_shape)

    def _postprocess_yolo11_raw(self, output: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        boxes_xywh = output[:, :4]
        class_scores = output[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.float32)
        confidences = np.max(class_scores, axis=1)
        keep = (class_ids == PERSON_CLASS_ID) & (confidences >= self.config.confidence_threshold)
        if not np.any(keep):
            return _empty_detections()

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
