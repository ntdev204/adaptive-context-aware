from __future__ import annotations

import os

from fastapi import FastAPI

from src.runtime.controller import JetsonRuntimeController, RuntimeConfig, RuntimeStatus

from .schemas import HealthResponse, RuntimeConfigResponse, RuntimeControlResponse, RuntimeMetricsResponse


def create_app(controller: JetsonRuntimeController | None = None) -> FastAPI:
    runtime = controller or JetsonRuntimeController(_load_runtime_config())
    app = FastAPI(
        title="Jetson Context Aware Control API",
        version="1.0.0",
        description="Control plane for the Jetson runtime. Frame and sensor data use ZMQ/protobuf data plane.",
    )
    app.state.runtime = runtime

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=RuntimeControlResponse)
    def ready() -> RuntimeControlResponse:
        return _control_response(runtime.status())

    @app.get("/config", response_model=RuntimeConfigResponse)
    def config() -> RuntimeConfigResponse:
        runtime_config = runtime.config
        jetson_host = runtime_config.jetson_host
        return RuntimeConfigResponse(
            jetson_host=jetson_host,
            pi_host=runtime_config.pi_host,
            sensor_ingest_endpoint=f"tcp://{jetson_host}:{runtime_config.sensor_ingest_port}",
            result_publish_endpoint=f"tcp://{jetson_host}:{runtime_config.result_publish_port}",
            heartbeat_endpoint=f"tcp://{runtime_config.pi_host}:{runtime_config.heartbeat_port}",
            engine_path=runtime_config.engine_path,
            camera_rgb_device=runtime_config.camera_rgb_device,
            camera_depth_device=runtime_config.camera_depth_device,
            max_sensor_age_ms=runtime_config.max_sensor_age_ms,
        )

    @app.get("/metrics", response_model=RuntimeMetricsResponse)
    def metrics() -> RuntimeMetricsResponse:
        status = runtime.status()
        ingest = status.ingest
        return RuntimeMetricsResponse(
            state=status.state.value,
            ready=status.ready,
            reason=status.reason,
            engine_available=status.engine_available,
            camera_available=status.camera_available,
            sensor_ingest_endpoint=ingest.endpoint,
            result_publish_endpoint=status.result_endpoint,
            heartbeat_endpoint=status.heartbeat_endpoint,
            messages_received=ingest.messages_received,
            decode_errors=ingest.decode_errors,
            last_message_age_ms=ingest.last_message_age_ms,
        )

    @app.post("/control/start", response_model=RuntimeControlResponse)
    def start_runtime() -> RuntimeControlResponse:
        return _control_response(runtime.start())

    @app.post("/control/stop", response_model=RuntimeControlResponse)
    def stop_runtime() -> RuntimeControlResponse:
        return _control_response(runtime.stop())

    return app


def _load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        jetson_host=os.environ.get("CTX_JETSON_HOST", "127.0.0.1"),
        pi_host=os.environ.get("CTX_PI_HOST", "25.12.4.101"),
        sensor_ingest_port=int(os.environ.get("CTX_SENSOR_INGEST_PORT", "5555")),
        result_publish_port=int(os.environ.get("CTX_RESULT_PUBLISH_PORT", "5556")),
        heartbeat_port=int(os.environ.get("CTX_HEARTBEAT_PORT", "9093")),
        heartbeat_interval_ms=int(os.environ.get("CTX_HEARTBEAT_INTERVAL_MS", "500")),
        max_sensor_age_ms=int(os.environ.get("CTX_MAX_SENSOR_AGE_MS", "250")),
        engine_path=os.environ.get("CTX_ENGINE_MODEL_PATH", "/app/models/engines/yolo11s.engine"),
        camera_rgb_device=os.environ.get("CTX_ASTRAS_RGB_DEVICE", "/dev/video0"),
        camera_depth_device=os.environ.get("CTX_ASTRAS_DEPTH_DEVICE", "/dev/video1"),
        camera_width=int(os.environ.get("CTX_ASTRAS_WIDTH", "640")),
        camera_height=int(os.environ.get("CTX_ASTRAS_HEIGHT", "480")),
        camera_fps=int(os.environ.get("CTX_ASTRAS_FPS", "30")),
    )


def _control_response(status: RuntimeStatus) -> RuntimeControlResponse:
    return RuntimeControlResponse(
        state=status.state.value,
        ready=status.ready,
        reason=status.reason,
    )


app = create_app()
