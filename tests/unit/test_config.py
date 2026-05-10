from __future__ import annotations

from config import load_config


def test_load_config() -> None:
    config = load_config("dev")
    assert config.environment == "dev"
    assert config.network.lidar_port == 9090
    assert config.safety.heartbeat_timeout_ms == 2000
