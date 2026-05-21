from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.perception.detector import DetectorConfig, PersonDetector, TensorRTInferenceUnavailableError


class FakeDetectorRuntime:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls = 0

    def run(self, input_batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert input_batch.shape == (1, 3, 480, 640)
        return self.output


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


def test_detector_engine_backend_runs_tensorrt_runtime_output(tmp_path) -> None:
    engine_path = tmp_path / "yolov8s.engine"
    engine_path.write_bytes(b"fake-engine")
    runtime = FakeDetectorRuntime(np.array([[100.0, 80.0, 40.0, 120.0, 0.91, 0.0]], dtype=np.float32))
    detector = PersonDetector(DetectorConfig(engine_path=engine_path), runtime=runtime)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame, frame_id=0)

    assert runtime.calls == 1
    assert result.backend == "engine"
    assert result.detections.shape == (1, 6)
    assert result.detections[0, 4] == pytest.approx(0.91)


def test_detector_engine_backend_decodes_yolov8_raw_output(tmp_path) -> None:
    engine_path = tmp_path / "yolov8s.engine"
    engine_path.write_bytes(b"fake-engine")
    # [B, channels, boxes] with [cx, cy, w, h, person_score, other_score]
    raw = np.array([[[120.0], [140.0], [40.0], [60.0], [0.93], [0.02]]], dtype=np.float32)
    detector = PersonDetector(DetectorConfig(engine_path=engine_path), runtime=FakeDetectorRuntime(raw))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(frame, frame_id=0)

    assert result.detections.shape == (1, 6)
    assert result.detections[0, :4].tolist() == pytest.approx([100.0, 110.0, 40.0, 60.0])
    assert result.detections[0, 5] == pytest.approx(0.0)


def test_detector_engine_backend_wraps_runtime_errors(tmp_path) -> None:
    class BadRuntime:
        def run(self, input_batch: np.ndarray) -> np.ndarray:
            del input_batch
            raise RuntimeError("cuda unavailable")

    engine_path = tmp_path / "yolov8s.engine"
    engine_path.write_bytes(b"fake-engine")
    detector = PersonDetector(DetectorConfig(engine_path=engine_path), runtime=BadRuntime())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(TensorRTInferenceUnavailableError, match="cuda unavailable"):
        detector.detect(frame, frame_id=0)


def test_detector_engine_backend_does_not_fallback_to_pt(tmp_path, monkeypatch) -> None:
    class BadRuntime:
        def run(self, input_batch: np.ndarray) -> np.ndarray:
            del input_batch
            raise RuntimeError("cuda unavailable")

    engine_path = tmp_path / "yolov8s.engine"
    pt_model_path = tmp_path / "best.pt"
    engine_path.write_bytes(b"fake-engine")
    pt_model_path.write_bytes(b"fake-weights")
    detector = PersonDetector(
        DetectorConfig(engine_path=engine_path, pt_model_path=pt_model_path),
        runtime=BadRuntime(),
    )
    fallback = np.array([[1.0, 2.0, 3.0, 4.0, 0.8, 0.0]], dtype=np.float32)
    monkeypatch.setattr(detector, "_infer_pt", lambda frame: fallback)

    with pytest.raises(TensorRTInferenceUnavailableError, match="cuda unavailable"):
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), frame_id=0)


def test_detector_raises_when_engine_missing() -> None:
    detector = PersonDetector(DetectorConfig(engine_path=Path("missing.engine")))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="missing TensorRT engine"):
        detector.detect(frame, frame_id=0)
