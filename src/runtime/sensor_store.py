from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic

from src.transport.messages import PiStatusMessage, SensorMessage


@dataclass(frozen=True, slots=True)
class SensorStoreSnapshot:
    pi_state: str | None
    pi_cpu_temp_c: float | None
    pi_cpu_load_pct: float | None
    last_pi_age_ms: float | None


class SensorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pi_status: PiStatusMessage | None = None
        self._pi_monotonic: float | None = None

    def update(self, message: SensorMessage) -> None:
        now = monotonic()
        with self._lock:
            if isinstance(message, PiStatusMessage):
                self._pi_status = message
                self._pi_monotonic = now

    def snapshot(self) -> SensorStoreSnapshot:
        with self._lock:
            now = monotonic()
            pi_age = None
            if self._pi_monotonic is not None:
                pi_age = (now - self._pi_monotonic) * 1000.0
            return SensorStoreSnapshot(
                pi_state=self._pi_status.state if self._pi_status else None,
                pi_cpu_temp_c=self._pi_status.cpu_temp_c if self._pi_status else None,
                pi_cpu_load_pct=self._pi_status.cpu_load_pct if self._pi_status else None,
                last_pi_age_ms=pi_age,
            )
