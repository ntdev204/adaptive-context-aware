from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

import zmq

from .messages import SensorMessage, SensorMessageCodec

SensorHandler = Callable[[SensorMessage], None]


@dataclass(frozen=True, slots=True)
class ZmqIngestConfig:
    bind_host: str = "25.12.4.100"
    bind_port: int = 5555
    recv_timeout_ms: int = 100
    high_water_mark: int = 1

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.bind_host}:{self.bind_port}"


@dataclass(frozen=True, slots=True)
class SensorIngestStats:
    running: bool
    endpoint: str
    messages_received: int
    decode_errors: int
    last_message_age_ms: float | None


class ZmqSensorIngest:
    """Adaptive runtime PULL endpoint for LiDAR/IMU packets pushed by Raspberry Pi."""

    def __init__(
        self,
        config: ZmqIngestConfig,
        handler: SensorHandler,
        context: zmq.Context | None = None,
    ) -> None:
        self.config = config
        self._handler = handler
        self._context = context or zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._messages_received = 0
        self._decode_errors = 0
        self._last_message_monotonic: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._socket.bind(self.config.endpoint)
        self._thread = threading.Thread(target=self._run, name="zmq-sensor-ingest", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        if self._socket:
            self._socket.close(linger=0)
            self._socket = None

    def stats(self) -> SensorIngestStats:
        with self._stats_lock:
            age_ms = None
            if self._last_message_monotonic is not None:
                age_ms = (monotonic() - self._last_message_monotonic) * 1000.0
            return SensorIngestStats(
                running=bool(self._thread and self._thread.is_alive()),
                endpoint=self.config.endpoint,
                messages_received=self._messages_received,
                decode_errors=self._decode_errors,
                last_message_age_ms=age_ms,
            )

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            try:
                message = SensorMessageCodec.decode(raw)
            except Exception:
                with self._stats_lock:
                    self._decode_errors += 1
                continue
            self._handler(message)
            with self._stats_lock:
                self._messages_received += 1
                self._last_message_monotonic = monotonic()
