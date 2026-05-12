from __future__ import annotations

from src.comm.health_monitor import HeartbeatWatchdog
from src.utils.enums import EStopReason, EStopSource, SafetyState


def test_heartbeat_loss_triggers_estop_within_timeout() -> None:
    now = {"value": 0}
    events: list[tuple[EStopReason, EStopSource]] = []
    watchdog = HeartbeatWatchdog(
        timeout_ms=2000,
        check_interval_ms=100,
        now_ms=lambda: now["value"],
        on_estop=lambda reason, source: events.append((reason, source)),
    )
    watchdog.record_heartbeat(at_ms=0)
    now["value"] = 1900
    assert watchdog.check() is False
    now["value"] = 2101
    assert watchdog.check() is True
    assert watchdog.state == SafetyState.ESTOP
    assert events == [(EStopReason.HEARTBEAT_TIMEOUT, EStopSource.RPI_WATCHDOG)]
