from __future__ import annotations

import numpy as np

from src.runtime.controller import JetsonRuntimeController, RuntimeConfig, RuntimeState
from src.runtime.perception_loop import PerceptionLoopStats
from src.transport.messages import ImuSampleMessage, LidarScanMessage


def test_runtime_status_requires_engine_camera_lidar_and_imu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CTX_CAMERA_BACKEND", "v4l2")
    engine_path = tmp_path / "yolov8s.engine"
    rgb_device = tmp_path / "astras-rgb"
    depth_device = tmp_path / "astras-depth"
    engine_path.write_bytes(b"engine")
    rgb_device.write_bytes(b"")
    depth_device.write_bytes(b"")

    controller = JetsonRuntimeController(
        RuntimeConfig(
            jetson_host="127.0.0.1",
            sensor_ingest_port=0,
            result_publish_port=0,
            engine_path=str(engine_path),
            camera_rgb_device=str(rgb_device),
            camera_depth_device=str(depth_device),
        )
    )
    controller.sensor_store.update(
        LidarScanMessage(
            source_id="pi-101",
            sequence=1,
            timestamp_us=100,
            scan_points=np.array([[0.0, 1.0], [0.1, 1.1]], dtype=np.float32),
        )
    )
    controller.sensor_store.update(
        ImuSampleMessage(
            source_id="pi-101",
            sequence=2,
            timestamp_us=101,
            accel_xyz_mps2=np.array([0.0, 0.0, 9.8], dtype=np.float32),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
    )

    status = controller.status()

    assert status.engine_available
    assert status.camera_available
    assert not status.ready
    assert status.state == "stopped"


def test_runtime_status_waits_for_perception_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CTX_CAMERA_BACKEND", "v4l2")
    engine_path = tmp_path / "best.engine"
    rgb_device = tmp_path / "astras-rgb"
    depth_device = tmp_path / "astras-depth"
    engine_path.write_bytes(b"engine")
    rgb_device.write_bytes(b"")
    depth_device.write_bytes(b"")

    controller = JetsonRuntimeController(
        RuntimeConfig(
            jetson_host="127.0.0.1",
            sensor_ingest_port=0,
            result_publish_port=0,
            engine_path=str(engine_path),
            camera_source="local-test",
            camera_rgb_device=str(rgb_device),
            camera_depth_device=str(depth_device),
        )
    )
    controller._state = RuntimeState.RUNNING
    controller._perception_loop = _FakePerceptionLoop(
        PerceptionLoopStats(
            running=True,
            frames_processed=0,
            publish_errors=0,
            last_result_age_ms=None,
            last_error=None,
        )
    )
    controller.sensor_store.update(
        LidarScanMessage(
            source_id="pi-101",
            sequence=1,
            timestamp_us=100,
            scan_points=np.array([[0.0, 1.0]], dtype=np.float32),
        )
    )
    controller.sensor_store.update(
        ImuSampleMessage(
            source_id="pi-101",
            sequence=2,
            timestamp_us=101,
            accel_xyz_mps2=np.array([0.0, 0.0, 9.8], dtype=np.float32),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
    )

    status = controller.status()

    assert not status.ready
    assert status.reason == "waiting for perception results"


class _FakePerceptionLoop:
    def __init__(self, stats: PerceptionLoopStats) -> None:
        self._stats = stats

    def stats(self) -> PerceptionLoopStats:
        return self._stats
