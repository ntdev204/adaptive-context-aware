from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class NetworkConfig(BaseModel):
    lidar_port: int = Field(ge=1, le=65535)
    nav_port: int = Field(ge=1, le=65535)
    soh_port: int = Field(ge=1, le=65535)
    heartbeat_port: int = Field(ge=1, le=65535)


class SafetyConfig(BaseModel):
    heartbeat_interval_ms: int = Field(gt=0)
    heartbeat_timeout_ms: int = Field(gt=0)
    heartbeat_check_interval_ms: int = Field(gt=0)


class LoggingConfig(BaseModel):
    level: str


class AppConfig(BaseModel):
    environment: Literal["dev", "test", "prod"]
    network: NetworkConfig
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
