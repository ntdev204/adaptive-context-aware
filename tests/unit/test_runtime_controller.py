from __future__ import annotations

import numpy as np

from src.runtime.controller import JetsonRuntimeController, RuntimeConfig
from src.transport.messages import ImuSampleMessage, LidarScanMessage


def test_runtime_status_requires_engine_camera_lidar_and_imu(tmp_path) -> None:
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


def test_runtime_start_enters_error_when_detector_warmup_fails(monkeypatch) -> None:
    import src.perception.pipeline as pipeline_module

    controller = JetsonRuntimeController(
        RuntimeConfig(
            jetson_host="127.0.0.1",
            sensor_ingest_port=0,
            result_publish_port=0,
            camera_source="disabled",
        )
    )

    class FakeService:
        def __init__(self, endpoint: str | None = None) -> None:
            self._stats = type("Stats", (), {"messages_received": 0, "decode_errors": 0, "last_message_age_ms": None})()
            self.config = type("Config", (), {"endpoint": endpoint or "tcp://127.0.0.1:0"})()

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def stats(self):
            return self._stats

    class FailingPipeline:
        def warmup(self) -> None:
            raise RuntimeError("bad engine")

    controller._frame_source = type("FrameSource", (), {"stop": lambda self: None})()
    controller._ingest = FakeService()
    controller._result_publisher = FakeService("tcp://127.0.0.1:5556")
    monkeypatch.setattr(controller, "_start_frame_source", lambda: None)
    monkeypatch.setattr(pipeline_module, "PerceptionPipeline", FailingPipeline)

    status = controller.start()

    assert status.state == "error"
    assert status.reason == "bad engine"
