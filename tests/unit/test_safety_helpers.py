from __future__ import annotations

import logging

from src.router.adaptive_router import ReasoningPathway
from src.safety.graceful_degrade import decide_degradation
from src.safety.logger import LoggerConfig, build_logger
from src.safety.metrics import SafetyMetrics
from src.safety.watchdog import ProcessWatchdog
from src.utils.enums import StatusChangeReason


def test_graceful_degradation_paths() -> None:
    oom = decide_degradation(gpu_oom=True)
    camera = decide_degradation(camera_failed=True)
    thermal = decide_degradation(gpu_temp_c=85.0)
    assert oom is not None and oom.reason == StatusChangeReason.SOH_LOW
    assert camera is not None and camera.reason == StatusChangeReason.CAMERA_FAIL
    assert thermal is not None and thermal.reason == StatusChangeReason.GPU_OVERHEAT


def test_metrics_payload_uses_pathway_names() -> None:
    metrics = SafetyMetrics(
        fps=30.0,
        total_latency_ms=20.0,
        detector_latency_ms=8.0,
        fusion_latency_ms=2.0,
        gpu_util_pct=55.0,
        cpu_util_pct=33.0,
        ram_used_mb=1024.0,
        pathway_selection={ReasoningPathway.GRU: 5},
    )
    payload = metrics.as_json_payload()
    assert payload["pathway_selection"] == {"gru": 5}


def test_logger_levels_follow_environment(tmp_path) -> None:
    dev_logger = build_logger(LoggerConfig(environment="dev", log_path=tmp_path / "dev.log"))
    prod_logger = build_logger(LoggerConfig(environment="prod", log_path=tmp_path / "prod.log"))
    assert dev_logger.level == logging.DEBUG
    assert prod_logger.level == logging.WARNING


def test_process_watchdog_requests_restart_after_timeout() -> None:
    clock = {"value": 0.0}
    watchdog = ProcessWatchdog(now_s=lambda: clock["value"], restart_timeout_s=5.0)
    watchdog.record_healthy(now_s=0.0)
    clock["value"] = 4.9
    assert watchdog.should_restart() is False
    clock["value"] = 5.1
    assert watchdog.should_restart() is True
