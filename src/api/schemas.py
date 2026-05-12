from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RuntimeConfigResponse(BaseModel):
    adaptive_host: str
    pi_host: str
    sensor_ingest_endpoint: str
    result_publish_endpoint: str
    engine_path: str
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
    sensor_ingest_endpoint: str
    result_publish_endpoint: str
    heartbeat_endpoint: str
    messages_received: int = Field(ge=0)
    decode_errors: int = Field(ge=0)
    last_message_age_ms: float | None = None
