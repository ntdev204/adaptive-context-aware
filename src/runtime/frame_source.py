from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import zmq


@dataclass(frozen=True, slots=True)
class CameraFrame:
    payload: bytes
    sequence: int
    timestamp_us: int
    received_monotonic: float


@dataclass(frozen=True, slots=True)
class ZmqJpegFrameConfig:
    host: str = "25.12.4.101"
    port: int = 5557
    recv_timeout_ms: int = 100
    high_water_mark: int = 2

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class FrameSourceStats:
    running: bool
    endpoint: str
    frames_received: int
    last_frame_age_ms: float | None


class ZmqJpegFrameReceiver:
    """Subscribe to the Wheeltec SCADA JPEG stream and keep the latest frame."""

    def __init__(self, config: ZmqJpegFrameConfig, context: zmq.Context | None = None) -> None:
        self.config = config
        self._context = context or zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._frames_received = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, self.config.high_water_mark)
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(self.config.endpoint)
        self._thread = threading.Thread(target=self._run, name="scada-jpeg-source", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        if self._socket:
            self._socket.close(linger=0)
            self._socket = None

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._latest

    def stats(self) -> FrameSourceStats:
        with self._lock:
            age_ms = None
            if self._latest is not None:
                age_ms = (time.monotonic() - self._latest.received_monotonic) * 1000.0
            return FrameSourceStats(
                running=bool(self._thread and self._thread.is_alive()),
                endpoint=self.config.endpoint,
                frames_received=self._frames_received,
                last_frame_age_ms=age_ms,
            )

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            if not _looks_like_jpeg(raw):
                continue
            now = time.monotonic()
            with self._lock:
                self._frames_received += 1
                self._latest = CameraFrame(
                    payload=bytes(raw),
                    sequence=self._frames_received,
                    timestamp_us=int(time.time() * 1_000_000),
                    received_monotonic=now,
                )


def _looks_like_jpeg(payload: bytes) -> bool:
    return len(payload) > 4 and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
