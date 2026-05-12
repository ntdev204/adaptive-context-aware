from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class NetworkConfig(BaseModel):
    adaptive_host: str
    jetson_host: str | None = None
    pi_host: str
    nav_port: int = Field(ge=1, le=65535)
    soh_port: int = Field(ge=1, le=65535)
    heartbeat_port: int = Field(ge=1, le=65535)
    sensor_ingest_port: int = Field(ge=1, le=65535)
    result_publish_port: int = Field(ge=1, le=65535)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_jetson_host(cls, value: object) -> object:
        if isinstance(value, dict) and "adaptive_host" not in value and "jetson_host" in value:
            value = {**value, "adaptive_host": value["jetson_host"]}
        return value

    @property
    def runtime_host(self) -> str:
        return self.adaptive_host


class SafetyConfig(BaseModel):
    heartbeat_interval_ms: int = Field(gt=0)
    heartbeat_timeout_ms: int = Field(gt=0)
    heartbeat_check_interval_ms: int = Field(gt=0)


class LoggingConfig(BaseModel):
    level: str


class CameraConfig(BaseModel):
    rgb_device: str
    depth_device: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)


class AppConfig(BaseModel):
    environment: Literal["dev", "test", "prod"]
    network: NetworkConfig
    camera: CameraConfig
    safety: SafetyConfig
    logging: LoggingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(environment: str, root: Path | None = None) -> AppConfig:
    config_root = root or Path(__file__).resolve().parent.parent / "config"
    with (config_root / "base.yaml").open("r", encoding="utf-8") as handle:
        base_payload = yaml.safe_load(handle) or {}
    with (config_root / f"{environment}.yaml").open("r", encoding="utf-8") as handle:
        env_payload = yaml.safe_load(handle) or {}
    payload = _deep_merge(base_payload, env_payload)
    return AppConfig.model_validate(payload)
