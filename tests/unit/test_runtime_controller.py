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


def test_runtime_start_warms_up_perception_before_camera_thread(monkeypatch) -> None:
    events: list[str] = []

    class FakeIngestService:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(f"start:{self.name}")

        def stop(self) -> None:
            events.append(f"stop:{self.name}")

        def stats(self):
            return type(
                "Stats",
                (),
                {"messages_received": 0, "decode_errors": 0, "last_message_age_ms": None},
            )()

    class FakePublisherService:
        def __init__(self, name: str) -> None:
            self.name = name
            self.config = type("Config", (), {"endpoint": "tcp://127.0.0.1:5556"})()

        def start(self) -> None:
            events.append(f"start:{self.name}")

        def stop(self) -> None:
            events.append(f"stop:{self.name}")

    def fail_perception_start() -> None:
        events.append("start:perception")
        raise RuntimeError("bad engine")

    controller = JetsonRuntimeController(
        RuntimeConfig(
            jetson_host="127.0.0.1",
            sensor_ingest_port=0,
            result_publish_port=0,
            camera_source="disabled",
        )
    )
    controller._ingest = FakeIngestService("ingest")
    controller._result_publisher = FakePublisherService("publisher")

    monkeypatch.setattr(controller, "_start_perception_loop", fail_perception_start)
    monkeypatch.setattr(controller, "_start_frame_source", lambda: events.append("start:camera"))
    monkeypatch.setattr(controller, "_start_heartbeat", lambda: events.append("start:heartbeat"))
    monkeypatch.setattr(controller, "_stop_perception_loop", lambda: events.append("stop:perception"))
    monkeypatch.setattr(controller, "_stop_frame_source", lambda: events.append("stop:camera"))
    monkeypatch.setattr(controller, "_stop_heartbeat", lambda: events.append("stop:heartbeat"))

    status = controller.start()

    assert status.state == "error"
    assert status.reason == "bad engine"
    assert "start:perception" in events
    assert "start:camera" not in events
