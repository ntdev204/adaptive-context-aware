from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable


@dataclass(slots=True)
class ProcessWatchdog:
    heartbeat_interval_s: float = 5.0
    restart_timeout_s: float = 5.0
    now_s: Callable[[], float] | None = None
    last_ok_s: float | None = None

    def _clock(self) -> float:
        return monotonic() if self.now_s is None else self.now_s()

    def record_healthy(self, now_s: float | None = None) -> None:
        self.last_ok_s = self._clock() if now_s is None else now_s

    def should_restart(self, now_s: float | None = None) -> bool:
        current = self._clock() if now_s is None else now_s
        if self.last_ok_s is None:
            self.last_ok_s = current
            return False
        return (current - self.last_ok_s) > self.restart_timeout_s
