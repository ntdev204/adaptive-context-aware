from __future__ import annotations

import time

from src.comm.health_monitor import HeartbeatServerDaemon, HeartbeatWatchdog
from src.utils.enums import EStopReason, EStopSource


def test_watchdog_daemon_triggers_without_heartbeat() -> None:
    events: list[tuple[EStopReason, EStopSource]] = []
    watchdog = HeartbeatWatchdog(
        timeout_ms=200,
        check_interval_ms=50,
        on_estop=lambda reason, source: events.append((reason, source)),
    )
    server = HeartbeatServerDaemon(host="127.0.0.1", port=0, watchdog=watchdog)
    server.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and not events:
            time.sleep(0.05)
        assert events == [(EStopReason.HEARTBEAT_TIMEOUT, EStopSource.RPI_WATCHDOG)]
    finally:
        server.stop()
