from __future__ import annotations

import numpy as np
import pytest

from src.perception.sensor_fusion import FusedEntity
from src.runtime.frame_source import LocalCameraFrameConfig, LocalCameraFrameSource, _looks_like_jpeg
from src.runtime.perception_loop import build_result_message


def test_build_result_message_maps_fused_entities_to_result_plane() -> None:
    entity = FusedEntity(
        track_id=7,
        bbox_xywh=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        position_3d=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        velocity_3d=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        heading_rad=0.25,
        confidence=0.91,
        nearest_obstacle_distance_m=1.4,
        nearest_obstacle_centroid_xy=np.array([0.0, 0.0], dtype=np.float32),
        ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
    )

    message = build_result_message(
        source_id="adaptive-runtime",
        sequence=11,
        timestamp_us=123456,
        entities=[entity],
        timings={"camera_ms": 2.0, "detector_ms": 5.0, "fusion_ms": 1.0, "total_ms": 10.0},
        actual_fps=30.0,
    )

    assert message.source_id == "adaptive-runtime"
    assert message.sequence == 11
    assert message.metrics.fps == pytest.approx(30.0)
    assert message.entities[0].track_id == 7
    np.testing.assert_allclose(message.entities[0].bbox_xywh, entity.bbox_xywh)
    np.testing.assert_allclose(message.entities[0].position_xyz_m, entity.position_3d)


def test_scada_frame_filter_accepts_only_jpeg_payloads() -> None:
    assert _looks_like_jpeg(b"\xff\xd8payload\xff\xd9")
    assert not _looks_like_jpeg(b"MAP:\x89PNG")
    assert not _looks_like_jpeg(b"not-a-frame")


def test_local_camera_defaults_to_openni() -> None:
    assert LocalCameraFrameConfig().backend == "openni"


def test_local_camera_openni_backend_uses_astra_openni_only(monkeypatch) -> None:
    source = LocalCameraFrameSource(LocalCameraFrameConfig(backend="openni"))
    fake_cv2 = _FakeCv2()

    monkeypatch.setattr(source, "_open_openni_camera", lambda: None)

    source._open_camera(fake_cv2)

    assert source._active_backend == "openni"
    assert source._capture is None
    assert fake_cv2.capture_calls == []


def test_local_camera_openni_backend_does_not_fallback_to_v4l2(monkeypatch) -> None:
    source = LocalCameraFrameSource(LocalCameraFrameConfig(backend="openni", rgb_device="/dev/video9"))
    fake_cv2 = _FakeCv2()

    def fail_openni() -> None:
        raise RuntimeError("openni unavailable")

    monkeypatch.setattr(source, "_open_openni_camera", fail_openni)

    with pytest.raises(RuntimeError, match="openni unavailable"):
        source._open_camera(fake_cv2)

    assert source._active_backend is None
    assert source._capture is None
    assert fake_cv2.capture_calls == []


class _FakeCapture:
    def __init__(self) -> None:
        self.settings: list[tuple[int, int]] = []

    def set(self, prop: int, value: int) -> None:
        self.settings.append((prop, value))

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_BUFFERSIZE = 4
    CAP_V4L2 = 200

    def __init__(self) -> None:
        self.capture_calls: list[tuple[str, int | None]] = []

    def VideoCapture(self, device: str, api_preference: int | None = None) -> _FakeCapture:  # noqa: N802
        self.capture_calls.append((device, api_preference))
        return _FakeCapture()
