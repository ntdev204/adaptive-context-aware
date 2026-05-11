from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LoggerConfig:
    environment: str
    log_path: Path


def build_logger(config: LoggerConfig) -> logging.Logger:
    logger = logging.getLogger(f"adaptive-context-aware.{config.environment}")
    logger.handlers.clear()
    logger.setLevel(_level_for_env(config.environment))
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(config.log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _level_for_env(environment: str) -> int:
    if environment == "dev":
        return logging.DEBUG
    if environment == "prod":
        return logging.WARNING
    return logging.INFO
