from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.runtime.controller import JetsonRuntimeController, RuntimeConfig, RuntimeState


def _client() -> TestClient:
    controller = JetsonRuntimeController(
        RuntimeConfig(
            bind_host="0.0.0.0",
            jetson_host="25.12.4.100",
            pi_host="127.0.0.2",
            sensor_ingest_port=0,
            result_publish_port=0,
            engine_path="/tmp/yolov8s.engine",
            camera_rgb_device="/tmp/astras-rgb",
            camera_depth_device="/tmp/astras-depth",
        )
    )
    return TestClient(create_app(controller))


def test_health_endpoint() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_control_plane_exposes_config_and_metrics() -> None:
    client = _client()
    config = client.get("/config")
    assert config.status_code == 200
    assert config.json()["bind_host"] == "0.0.0.0"
    assert config.json()["jetson_host"] == "25.12.4.100"
    assert config.json()["sensor_ingest_endpoint"] == "tcp://25.12.4.100:0"
    assert config.json()["result_publish_endpoint"] == "tcp://25.12.4.100:0"
    assert config.json()["heartbeat_endpoint"] == "tcp://127.0.0.2:9093"
    assert config.json()["camera_rgb_device"] == "/tmp/astras-rgb"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["state"] == "stopped"
    assert payload["engine_available"] is False
    assert payload["camera_available"] is False
    assert payload["sensor_ingest_endpoint"] == "tcp://25.12.4.100:0"
    assert payload["result_publish_endpoint"] == "tcp://25.12.4.100:0"
    assert payload["heartbeat_endpoint"] == "tcp://127.0.0.2:9093"
    assert payload["messages_received"] == 0


def test_http_frame_endpoint_is_not_available() -> None:
    response = _client().post("/v1/perception/frame")
    assert response.status_code == 404


def test_autostart_raises_when_runtime_start_fails(monkeypatch) -> None:
    monkeypatch.setenv("CTX_AUTOSTART", "1")

    class FailingController:
        def start(self):
            return SimpleNamespace(state=RuntimeState.ERROR, reason="bad engine")

        def stop(self):
            return None

    with pytest.raises(RuntimeError, match="bad engine"):
        with TestClient(create_app(FailingController())):
            pass
