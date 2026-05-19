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
class LocalCameraFrameConfig:
    backend: str = "openni"
    rgb_device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    read_interval_ms: int = 33
    publish_host: str = "0.0.0.0"
    publish_port: int = 5557
    publish_enabled: bool = True
    jpeg_quality: int = 80

    @property
    def endpoint(self) -> str:
        if not self.publish_enabled:
            return f"local://{self.backend}"
        return f"tcp://{self.publish_host}:{self.publish_port}"


@dataclass(frozen=True, slots=True)
class FrameSourceStats:
    running: bool
    endpoint: str
    frames_received: int
    last_frame_age_ms: float | None
    last_error: str | None = None


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
                last_error=None,
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


class LocalCameraFrameSource:
    """Capture a local Jetson camera and optionally publish JPEG frames for the dashboard."""

    def __init__(self, config: LocalCameraFrameConfig, context: zmq.Context | None = None) -> None:
        self.config = config
        self._context = context or zmq.Context.instance()
        self._publisher: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._frames_received = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        if self.config.publish_enabled:
            self._publisher = self._context.socket(zmq.PUB)
            self._publisher.setsockopt(zmq.SNDHWM, 2)
            self._publisher.setsockopt(zmq.LINGER, 0)
            self._publisher.bind(self.config.endpoint)
        self._thread = threading.Thread(target=self._run, name="local-camera-source", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        if self._publisher:
            self._publisher.close(linger=0)
            self._publisher = None

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
                last_error=self._last_error,
            )

    def _run(self) -> None:
        import cv2

        capture = self._open_capture(cv2)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"failed to open local camera backend={self.config.backend}")
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
            interval_s = max(self.config.read_interval_ms, 1) / 1000.0
            while not self._stop_event.is_set():
                ok, frame = self._read_frame(cv2, capture)
                if not ok or frame is None:
                    with self._lock:
                        self._last_error = f"camera read failed backend={self.config.backend}"
                    time.sleep(interval_s)
                    continue

                ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    with self._lock:
                        self._last_error = "failed to encode camera frame as JPEG"
                    time.sleep(interval_s)
                    continue

                payload = encoded.tobytes()
                now = time.monotonic()
                with self._lock:
                    self._frames_received += 1
                    camera_frame = CameraFrame(
                        payload=payload,
                        sequence=self._frames_received,
                        timestamp_us=int(time.time() * 1_000_000),
                        received_monotonic=now,
                    )
                    self._latest = camera_frame
                    self._last_error = None

                if self._publisher is not None:
                    self._publisher.send(payload)
                time.sleep(interval_s)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            capture.release()

    def _open_capture(self, cv2):
        if self.config.backend == "openni":
            api_preference = getattr(cv2, "CAP_OPENNI2", 0)
            capture = cv2.VideoCapture(api_preference)
            if capture.isOpened():
                return capture
            capture.release()
            with self._lock:
                self._last_error = "OpenNI backend did not open; falling back to RGB device capture"
            capture = cv2.VideoCapture(self.config.rgb_device or "/dev/video0")
        else:
            capture = cv2.VideoCapture(self.config.rgb_device or "/dev/video0")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        return capture

    def _read_frame(self, cv2, capture):
        if self.config.backend == "openni":
            if not capture.grab():
                return False, None
            ok, frame = capture.retrieve(None, getattr(cv2, "CAP_OPENNI_BGR_IMAGE", 5))
            return ok, frame
        return capture.read()


def _looks_like_jpeg(payload: bytes) -> bool:
    return len(payload) > 4 and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
