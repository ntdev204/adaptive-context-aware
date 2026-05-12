from __future__ import annotations

from config import load_config


def test_load_config() -> None:
    config = load_config("dev")
    assert config.environment == "dev"
    assert config.network.jetson_host == "25.12.4.100"
    assert config.network.pi_host == "25.12.4.101"
    assert config.network.lidar_port == 9090
    assert config.network.sensor_ingest_port == 5555
    assert config.network.result_publish_port == 5556
    assert config.camera.rgb_device == "/dev/video0"
    assert config.camera.depth_device == "/dev/video1"
    assert config.safety.heartbeat_timeout_ms == 2000
