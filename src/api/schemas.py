from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RuntimeConfigResponse(BaseModel):
    bind_host: str
    jetson_host: str
    pi_host: str
    sensor_ingest_endpoint: str
    result_publish_endpoint: str
    engine_path: str
    camera_source: str
    scada_camera_endpoint: str | None = None
    camera_rgb_device: str
    camera_depth_device: str
    max_sensor_age_ms: int
    heartbeat_endpoint: str


class RuntimeControlResponse(BaseModel):
    state: str
    ready: bool
    reason: str | None = None


class RuntimeMetricsResponse(BaseModel):
    state: str
    ready: bool
    reason: str | None = None
    engine_available: bool
    camera_available: bool
    perception_running: bool
    frames_processed: int = Field(ge=0)
    last_result_age_ms: float | None = None
    last_runtime_error: str | None = None
    sensor_ingest_endpoint: str
    result_publish_endpoint: str
    heartbeat_endpoint: str
    messages_received: int = Field(ge=0)
    decode_errors: int = Field(ge=0)
    last_message_age_ms: float | None = None
