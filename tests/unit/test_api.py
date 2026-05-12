from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.runtime.controller import AdaptiveRuntimeController, RuntimeConfig


def _client() -> TestClient:
    controller = AdaptiveRuntimeController(
        RuntimeConfig(
            adaptive_host="127.0.0.1",
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
    assert config.json()["adaptive_host"] == "127.0.0.1"
    assert config.json()["sensor_ingest_endpoint"] == "tcp://127.0.0.1:0"
    assert config.json()["result_publish_endpoint"] == "tcp://127.0.0.1:0"
    assert config.json()["heartbeat_endpoint"] == "tcp://127.0.0.2:9093"
    assert config.json()["camera_rgb_device"] == "/tmp/astras-rgb"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["state"] == "stopped"
    assert payload["engine_available"] is False
    assert payload["camera_available"] is False
    assert payload["result_publish_endpoint"] == "tcp://127.0.0.1:0"
    assert payload["heartbeat_endpoint"] == "tcp://127.0.0.2:9093"
    assert payload["messages_received"] == 0


def test_http_frame_endpoint_is_not_available() -> None:
    response = _client().post("/v1/perception/frame")
    assert response.status_code == 404
