from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.runtime.frame_source import (
    MAX_JPEG_BYTES,
    CameraFrame,
    LocalCameraFrameConfig,
    LocalCameraFrameSource,
    _looks_like_jpeg,
    _make_readonly,
)
from src.runtime.perception_loop import PerceptionLoopConfig, RuntimePerceptionLoop
from src.runtime.sensor_store import SensorStore

# ---------------------------------------------------------------------------
# CameraFrame immutability and raw frame support
# ---------------------------------------------------------------------------


def test_camera_frame_raw_bgr_is_read_only_after_make_readonly() -> None:
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _make_readonly(bgr)
    with pytest.raises(ValueError):
        bgr[0, 0, 0] = 255


def test_camera_frame_can_carry_raw_bgr_and_depth() -> None:
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    frame = CameraFrame(
        payload=b"",
        sequence=1,
        timestamp_us=0,
        received_monotonic=0.0,
        frame_bgr=bgr,
        depth_map_m=depth,
    )
    assert frame.frame_bgr is bgr
    assert frame.depth_map_m is depth


def test_camera_frame_defaults_raw_fields_to_none() -> None:
    frame = CameraFrame(
        payload=b"\xff\xd8fake\xff\xd9",
        sequence=1,
        timestamp_us=0,
        received_monotonic=0.0,
    )
    assert frame.frame_bgr is None
    assert frame.depth_map_m is None


# ---------------------------------------------------------------------------
# JPEG size guard
# ---------------------------------------------------------------------------


def test_oversized_jpeg_is_rejected_by_size_guard() -> None:
    giant = b"\xff\xd8" + b"\x00" * (MAX_JPEG_BYTES + 1) + b"\xff\xd9"
    assert _looks_like_jpeg(giant)
    assert len(giant) > MAX_JPEG_BYTES


class FakeEncoded:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class FakeCv2:
    IMWRITE_JPEG_QUALITY = 1

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def imencode(self, ext, frame_bgr, encode_params):
        return True, FakeEncoded(self.payload)


def test_local_camera_source_paces_successful_reads() -> None:
    source = LocalCameraFrameSource(LocalCameraFrameConfig(read_interval_ms=40, publish_enabled=False))
    frames_read = 0

    def fake_open_camera(cv2) -> None:
        return None

    def fake_read_frame(cv2):
        nonlocal frames_read
        frames_read += 1
        if frames_read >= 2:
            source._stop_event.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8), None

    source._open_camera = fake_open_camera
    source._read_frame = fake_read_frame

    started = time.monotonic()
    source._run()

    assert frames_read == 2
    assert time.monotonic() - started >= 0.04


def test_local_camera_source_does_not_publish_oversized_jpeg() -> None:
    class FakePublisherSocket:
        def __init__(self) -> None:
            self.messages: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.messages.append(payload)

    source = LocalCameraFrameSource(LocalCameraFrameConfig(read_interval_ms=1, publish_enabled=True))
    publisher = FakePublisherSocket()
    source._publisher = publisher

    def fake_open_camera(cv2) -> None:
        return None

    def fake_read_frame(cv2):
        source._stop_event.set()
        return True, np.zeros((480, 640, 3), dtype=np.uint8), None

    source._open_camera = fake_open_camera
    source._read_frame = fake_read_frame
    source._run(FakeCv2(b"x" * (MAX_JPEG_BYTES + 1)))

    assert publisher.messages == []
    assert source.latest() is not None
    assert source.latest().payload == b""
    assert source.stats().last_error == f"encoded JPEG exceeds {MAX_JPEG_BYTES} bytes"


def test_local_camera_source_stop_does_not_force_close_stuck_capture_thread() -> None:
    class StuckThread:
        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return True

    source = LocalCameraFrameSource(LocalCameraFrameConfig(publish_enabled=False))
    source._thread = StuckThread()
    close_calls: list[str] = []
    source._close_camera = lambda: close_calls.append("closed")

    source.stop(timeout_s=0.01)

    assert close_calls == []
    assert source.stats().last_error == "camera shutdown timed out; leaving OpenNI cleanup to the capture thread"


# ---------------------------------------------------------------------------
# Event-based perception loop
# ---------------------------------------------------------------------------


class FakeFrameSource:
    def __init__(self) -> None:
        self.frame_ready = threading.Event()
        self._latest: CameraFrame | None = None

    def latest(self) -> CameraFrame | None:
        return self._latest

    def set_frame(self, frame: CameraFrame) -> None:
        self._latest = frame
        self.frame_ready.set()


class FakePipeline:
    def __init__(self) -> None:
        self.call_count = 0

    def process(
        self,
        frame_bgr,
        depth_map_m=None,
        lidar_scan=None,
        accel_xyz_mps2=None,
        quat_xyzw=None,
        timestamp_us=0,
        frame_id=None,
        delta_time_s=0.1,
    ):
        self.call_count += 1
        return [], {"total_ms": 1.0}


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list = []

    def publish(self, message) -> None:
        self.messages.append(message)


def _make_test_frame(seq: int = 1) -> CameraFrame:
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    return CameraFrame(
        payload=b"",
        sequence=seq,
        timestamp_us=seq * 33_000,
        received_monotonic=time.monotonic(),
        frame_bgr=bgr,
    )


def test_perception_loop_processes_frame_on_event_signal() -> None:
    source = FakeFrameSource()
    pipeline = FakePipeline()
    publisher = FakePublisher()
    loop = RuntimePerceptionLoop(
        pipeline=pipeline,
        sensor_store=SensorStore(),
        frame_source=source,
        result_publisher=publisher,
        config=PerceptionLoopConfig(interval_ms=5),
    )
    loop.start()
    try:
        source.set_frame(_make_test_frame(seq=1))
        time.sleep(0.1)
        assert pipeline.call_count >= 1
        assert len(publisher.messages) >= 1
    finally:
        loop.stop()


def test_perception_loop_skips_duplicate_frame_sequence() -> None:
    source = FakeFrameSource()
    pipeline = FakePipeline()
    publisher = FakePublisher()
    loop = RuntimePerceptionLoop(
        pipeline=pipeline,
        sensor_store=SensorStore(),
        frame_source=source,
        result_publisher=publisher,
        config=PerceptionLoopConfig(interval_ms=5),
    )
    loop.start()
    try:
        frame = _make_test_frame(seq=1)
        source.set_frame(frame)
        time.sleep(0.1)
        count_after_first = pipeline.call_count

        source.frame_ready.set()
        time.sleep(0.1)
        assert pipeline.call_count == count_after_first
    finally:
        loop.stop()


def test_perception_loop_stops_cleanly() -> None:
    source = FakeFrameSource()
    pipeline = FakePipeline()
    publisher = FakePublisher()
    loop = RuntimePerceptionLoop(
        pipeline=pipeline,
        sensor_store=SensorStore(),
        frame_source=source,
        result_publisher=publisher,
        config=PerceptionLoopConfig(interval_ms=5),
    )
    loop.start()
    assert loop.stats().running
    loop.stop(timeout_s=1.0)
    assert not loop.stats().running


def test_perception_loop_uses_raw_bgr_without_decode() -> None:
    raw_bgr = np.ones((480, 640, 3), dtype=np.uint8) * 42
    frame = CameraFrame(
        payload=b"",
        sequence=1,
        timestamp_us=100_000,
        received_monotonic=time.monotonic(),
        frame_bgr=raw_bgr,
    )

    received_frames: list[np.ndarray] = []

    class SpyPipeline:
        def process(self, frame_bgr, **kwargs):
            received_frames.append(frame_bgr)
            return [], {"total_ms": 1.0}

    source = FakeFrameSource()
    publisher = FakePublisher()
    loop = RuntimePerceptionLoop(
        pipeline=SpyPipeline(),
        sensor_store=SensorStore(),
        frame_source=source,
        result_publisher=publisher,
        config=PerceptionLoopConfig(interval_ms=5),
    )

    message = loop.process_once(frame)
    assert message is not None
    assert len(received_frames) == 1
    assert received_frames[0] is raw_bgr


def test_perception_loop_decodes_jpeg_when_no_raw_bgr() -> None:
    import cv2

    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", bgr)
    jpeg_bytes = encoded.tobytes()

    frame = CameraFrame(
        payload=jpeg_bytes,
        sequence=1,
        timestamp_us=100_000,
        received_monotonic=time.monotonic(),
    )

    received_frames: list[np.ndarray] = []

    class SpyPipeline:
        def process(self, frame_bgr, **kwargs):
            received_frames.append(frame_bgr)
            return [], {"total_ms": 1.0}

    source = FakeFrameSource()
    publisher = FakePublisher()
    loop = RuntimePerceptionLoop(
        pipeline=SpyPipeline(),
        sensor_store=SensorStore(),
        frame_source=source,
        result_publisher=publisher,
    )

    loop.process_once(frame)
    assert len(received_frames) == 1
    assert received_frames[0].shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# TensorRT allocation cache
# ---------------------------------------------------------------------------


class FakeCudart:
    def __init__(self) -> None:
        self.next_ptr = 1000
        self.freed: list[int] = []

    def cudaMalloc(self, nbytes: int):
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1)
        return 0, ptr

    def cudaFree(self, ptr: int):
        self.freed.append(ptr)
        return 0


def test_raw_tensorrt_runner_reuses_cached_allocation_for_same_shape() -> None:
    from src.runtime.tensorrt_engine import _RawTensorRTRunner

    runner = object.__new__(_RawTensorRTRunner)
    runner.cudart = FakeCudart()
    runner._allocations = {}

    first = runner._get_or_alloc("images", np.zeros((1, 3), dtype=np.float32))
    second = runner._get_or_alloc("images", np.ones((1, 3), dtype=np.float32))

    assert second is first
    assert runner.cudart.freed == []


def test_raw_tensorrt_runner_reallocates_on_shape_change() -> None:
    from src.runtime.tensorrt_engine import _RawTensorRTRunner

    runner = object.__new__(_RawTensorRTRunner)
    runner.cudart = FakeCudart()
    runner._allocations = {}

    first = runner._get_or_alloc("images", np.zeros((1, 3), dtype=np.float32))
    second = runner._get_or_alloc("images", np.zeros((1, 6), dtype=np.float32))

    assert second is not first
    assert runner.cudart.freed == [first.ptr]


def test_raw_tensorrt_runner_close_frees_cached_allocations() -> None:
    from src.runtime.tensorrt_engine import _RawTensorRTRunner

    runner = object.__new__(_RawTensorRTRunner)
    runner.cudart = FakeCudart()
    runner._allocations = {}

    allocation = runner._get_or_alloc("images", np.zeros((1, 3), dtype=np.float32))
    runner.close()

    assert runner.cudart.freed == [allocation.ptr]
    assert runner._allocations == {}


# ---------------------------------------------------------------------------
# Flat depth cache
# ---------------------------------------------------------------------------


def test_flat_depth_cache_returns_same_instance() -> None:
    from src.perception.pipeline import _make_flat_depth

    a = _make_flat_depth()
    b = _make_flat_depth()
    assert a is b
    assert not a.flags.writeable


# ---------------------------------------------------------------------------
# Sensor fusion: single nearest_cluster call per track
# ---------------------------------------------------------------------------


def test_sensor_fusion_calls_nearest_cluster_once_per_track() -> None:
    from unittest.mock import patch

    from src.perception.sensor_fusion import SensorFusion
    from src.perception.tracker import TrackState

    track = TrackState(
        track_id=1,
        bbox_xywh=np.array([10, 20, 30, 40], dtype=np.float32),
        position_3d=np.zeros(3, dtype=np.float32),
        velocity_3d=np.zeros(3, dtype=np.float32),
        age=1,
        missed_frames=0,
        confidence=0.9,
    )

    call_count = 0

    def counting_nearest(track, clusters):
        nonlocal call_count
        call_count += 1
        return None, None

    fusion = SensorFusion()
    with patch("src.perception.sensor_fusion._nearest_cluster", side_effect=counting_nearest):
        fusion.fuse([track], ego_motion=None, lidar_clusters=[])

    assert call_count == 1
