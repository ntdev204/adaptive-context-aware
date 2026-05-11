from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.perception.detector import DetectorConfig, PersonDetector, TensorRTInferenceUnavailableError


def test_detector_preprocess_matches_yolo_contract() -> None:
    detector = PersonDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor = detector.preprocess(frame)
    assert tensor.shape == (1, 3, 480, 640)
    assert tensor.dtype == np.float32


def test_detector_reads_fixture_annotation_as_detection() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "annotations"
    detector = PersonDetector(DetectorConfig(backend="synthetic", annotation_dir=root))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame, frame_id=0)
    assert result.backend == "synthetic"
    assert result.detections.shape == (1, 6)
    assert result.detections[0, 4] == pytest.approx(0.92)
    assert result.detections[0, 5] == pytest.approx(0.0)


def test_detector_returns_empty_array_when_fixture_missing() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "annotations"
    detector = PersonDetector(DetectorConfig(backend="synthetic", annotation_dir=root))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame, frame_id=99)
    assert result.detections.shape == (0, 6)


def test_detector_rejects_wrong_frame_shape() -> None:
    detector = PersonDetector()
    frame = np.zeros((320, 240, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected BGR frame"):
        detector.preprocess(frame)


def test_detector_engine_path_prefers_real_engine(tmp_path) -> None:
    engine_path = tmp_path / "yolov8s.engine"
    engine_path.write_bytes(b"fake-engine")
    detector = PersonDetector(DetectorConfig(engine_path=engine_path))
    assert detector._engine_path() == engine_path


def test_detector_engine_backend_fails_until_tensorrt_bindings_exist(tmp_path) -> None:
    engine_path = tmp_path / "yolov8s.engine"
    engine_path.write_bytes(b"fake-engine")
    detector = PersonDetector(DetectorConfig(engine_path=engine_path))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(TensorRTInferenceUnavailableError):
        detector.detect(frame, frame_id=0)


def test_detector_raises_when_engine_missing() -> None:
    detector = PersonDetector(DetectorConfig(engine_path=Path("missing.engine")))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="missing TensorRT engine"):
        detector.detect(frame, frame_id=0)
