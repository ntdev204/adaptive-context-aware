from __future__ import annotations

from src.runtime.controller import AdaptiveRuntimeController, RuntimeConfig


def test_runtime_status_requires_engine_and_camera(tmp_path) -> None:
    engine_path = tmp_path / "yolov8s.engine"
    rgb_device = tmp_path / "astras-rgb"
    depth_device = tmp_path / "astras-depth"
    engine_path.write_bytes(b"engine")
    rgb_device.write_bytes(b"")
    depth_device.write_bytes(b"")

    controller = AdaptiveRuntimeController(
        RuntimeConfig(
            adaptive_host="127.0.0.1",
            sensor_ingest_port=0,
            result_publish_port=0,
            engine_path=str(engine_path),
            camera_rgb_device=str(rgb_device),
            camera_depth_device=str(depth_device),
        )
    )

    status = controller.status()

    assert status.engine_available
    assert status.camera_available
    assert not status.ready
    assert status.state == "stopped"
