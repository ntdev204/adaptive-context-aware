from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from time import monotonic

from src.transport.messages import ImuSampleMessage, LidarScanMessage, PiStatusMessage, SensorMessage


@dataclass(frozen=True, slots=True)
class SensorStoreSnapshot:
    has_lidar: bool
    has_imu: bool
    last_lidar_age_ms: float | None
    last_imu_age_ms: float | None
    pi_state: str | None
    pi_cpu_temp_c: float | None
    pi_cpu_load_pct: float | None


class SensorStore:
    def __init__(self, history_size: int = 32) -> None:
        self._lock = threading.Lock()
        self._lidar: LidarScanMessage | None = None
        self._imu: ImuSampleMessage | None = None
        self._pi_status: PiStatusMessage | None = None
        self._lidar_monotonic: float | None = None
        self._imu_monotonic: float | None = None
        self._lidar_history: deque[LidarScanMessage] = deque(maxlen=history_size)
        self._imu_history: deque[ImuSampleMessage] = deque(maxlen=history_size)

    def update(self, message: SensorMessage) -> None:
        now = monotonic()
        with self._lock:
            if isinstance(message, LidarScanMessage):
                self._lidar = message
                self._lidar_monotonic = now
                self._lidar_history.append(message)
            elif isinstance(message, ImuSampleMessage):
                self._imu = message
                self._imu_monotonic = now
                self._imu_history.append(message)
            elif isinstance(message, PiStatusMessage):
                self._pi_status = message

    def latest_lidar(self) -> LidarScanMessage | None:
        with self._lock:
            return self._lidar

    def latest_imu(self) -> ImuSampleMessage | None:
        with self._lock:
            return self._imu

    def lidar_nearest(self, timestamp_us: int) -> LidarScanMessage | None:
        with self._lock:
            return _nearest_by_timestamp(self._lidar_history, timestamp_us)

    def imu_nearest(self, timestamp_us: int) -> ImuSampleMessage | None:
        with self._lock:
            return _nearest_by_timestamp(self._imu_history, timestamp_us)

    def snapshot(self) -> SensorStoreSnapshot:
        with self._lock:
            now = monotonic()
            lidar_age = None
            imu_age = None
            if self._lidar_monotonic is not None:
                lidar_age = (now - self._lidar_monotonic) * 1000.0
            if self._imu_monotonic is not None:
                imu_age = (now - self._imu_monotonic) * 1000.0
            return SensorStoreSnapshot(
                has_lidar=self._lidar is not None,
                has_imu=self._imu is not None,
                last_lidar_age_ms=lidar_age,
                last_imu_age_ms=imu_age,
                pi_state=self._pi_status.state if self._pi_status else None,
                pi_cpu_temp_c=self._pi_status.cpu_temp_c if self._pi_status else None,
                pi_cpu_load_pct=self._pi_status.cpu_load_pct if self._pi_status else None,
            )


def _nearest_by_timestamp(messages, timestamp_us: int):
    if not messages:
        return None
    return min(messages, key=lambda message: abs(message.timestamp_us - timestamp_us))
