from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.runtime.controller import JetsonRuntimeController, RuntimeConfig, RuntimeStatus

from .schemas import HealthResponse, RuntimeConfigResponse, RuntimeControlResponse, RuntimeMetricsResponse


def create_app(controller: JetsonRuntimeController | None = None) -> FastAPI:
    runtime = controller or JetsonRuntimeController(_load_runtime_config())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if _env_bool("CTX_AUTOSTART", False):
            runtime.start()
        try:
            yield
        finally:
            runtime.stop()

    app = FastAPI(
        title="Jetson Context Aware Control API",
        version="1.0.1",
        description="Control plane for the Jetson runtime. Frame and sensor data use ZMQ/protobuf data plane.",
        lifespan=lifespan,
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
        jetson_host = _public_jetson_host(runtime_config)
        scada_camera_endpoint = None
        if runtime_config.camera_source == "scada_zmq":
            scada_camera_endpoint = f"tcp://{runtime_config.scada_camera_host}:{runtime_config.scada_camera_port}"
        return RuntimeConfigResponse(
            bind_host=runtime_config.bind_host or runtime_config.jetson_host,
            jetson_host=jetson_host,
            pi_host=runtime_config.pi_host,
            sensor_ingest_endpoint=f"tcp://{jetson_host}:{runtime_config.sensor_ingest_port}",
            result_publish_endpoint=f"tcp://{jetson_host}:{runtime_config.result_publish_port}",
            heartbeat_endpoint=f"tcp://{runtime_config.pi_host}:{runtime_config.heartbeat_port}",
            engine_path=runtime_config.engine_path,
            camera_source=runtime_config.camera_source,
            scada_camera_endpoint=scada_camera_endpoint,
            camera_rgb_device=runtime_config.camera_rgb_device,
            camera_depth_device=runtime_config.camera_depth_device,
            max_sensor_age_ms=runtime_config.max_sensor_age_ms,
        )

    @app.get("/metrics", response_model=RuntimeMetricsResponse)
    def metrics() -> RuntimeMetricsResponse:
        status = runtime.status()
        ingest = status.ingest
        jetson_host = _public_jetson_host(runtime.config)
        return RuntimeMetricsResponse(
            state=status.state.value,
            ready=status.ready,
            reason=status.reason,
            engine_available=status.engine_available,
            camera_available=status.camera_available,
            perception_running=status.perception_running,
            frames_processed=status.frames_processed,
            last_result_age_ms=status.last_result_age_ms,
            last_runtime_error=status.last_runtime_error,
            sensor_ingest_endpoint=f"tcp://{jetson_host}:{runtime.config.sensor_ingest_port}",
            result_publish_endpoint=f"tcp://{jetson_host}:{runtime.config.result_publish_port}",
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
        bind_host=os.environ.get("CTX_BIND_HOST"),
        jetson_host=os.environ.get("CTX_JETSON_HOST", "25.12.4.100"),
        pi_host=os.environ.get("CTX_PI_HOST", "25.12.4.101"),
        runtime_backend=os.environ.get("CTX_RUNTIME_BACKEND", "engine"),
        perception_enabled=_env_bool("CTX_PERCEPTION_ENABLED", True),
        perception_interval_ms=int(os.environ.get("CTX_PERCEPTION_INTERVAL_MS", "5")),
        result_source_id=os.environ.get("CTX_RESULT_SOURCE_ID", "adaptive-runtime"),
        sensor_ingest_port=int(os.environ.get("CTX_SENSOR_INGEST_PORT", "5555")),
        result_publish_port=int(os.environ.get("CTX_RESULT_PUBLISH_PORT", "5556")),
        heartbeat_port=int(os.environ.get("CTX_HEARTBEAT_PORT", "9093")),
        heartbeat_interval_ms=int(os.environ.get("CTX_HEARTBEAT_INTERVAL_MS", "500")),
        max_sensor_age_ms=int(os.environ.get("CTX_MAX_SENSOR_AGE_MS", "250")),
        engine_path=os.environ.get("CTX_ENGINE_MODEL_PATH", "/app/models/engines/best.engine"),
        pt_model_path=os.environ.get("CTX_PT_MODEL_PATH", "/app/models/fine_tuning/best.pt"),
        camera_source=os.environ.get("CTX_CAMERA_SOURCE", "device").strip().lower(),
        scada_camera_host=os.environ.get("CTX_SCADA_CAMERA_HOST", os.environ.get("CTX_PI_HOST", "25.12.4.101")),
        scada_camera_port=int(os.environ.get("CTX_SCADA_CAMERA_PORT", "5557")),
        camera_frame_timeout_ms=int(os.environ.get("CTX_CAMERA_FRAME_TIMEOUT_MS", "2000")),
        camera_rgb_device=os.environ.get("CTX_ASTRAS_RGB_DEVICE", "/dev/video0"),
        camera_depth_device=os.environ.get("CTX_ASTRAS_DEPTH_DEVICE", "/dev/video1"),
        camera_width=int(os.environ.get("CTX_ASTRAS_WIDTH", "640")),
        camera_height=int(os.environ.get("CTX_ASTRAS_HEIGHT", "480")),
        camera_fps=int(os.environ.get("CTX_ASTRAS_FPS", "30")),
        camera_publish_port=int(os.environ.get("CTX_CAMERA_PUBLISH_PORT", "5557")),
    )


def _control_response(status: RuntimeStatus) -> RuntimeControlResponse:
    return RuntimeControlResponse(
        state=status.state.value,
        ready=status.ready,
        reason=status.reason,
    )


def _public_jetson_host(config: RuntimeConfig) -> str:
    return config.jetson_host


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


app = create_app()
