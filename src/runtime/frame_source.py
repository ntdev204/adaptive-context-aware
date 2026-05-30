from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq

log = logging.getLogger(__name__)


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
    error_log_interval_s: float = 5.0

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
        self._capture = None
        self._openni2 = None
        self._oni_device = None
        self._oni_depth_stream = None
        self._oni_color_stream = None
        self._active_backend: str | None = None
        self._last_error_logged_monotonic = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        log.info(
            "starting local camera source backend=%s rgb_device=%s resolution=%sx%s fps=%s publish=%s endpoint=%s",
            self.config.backend,
            self.config.rgb_device,
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.publish_enabled,
            self.config.endpoint,
        )
        if self.config.publish_enabled:
            self._publisher = self._context.socket(zmq.PUB)
            self._publisher.setsockopt(zmq.SNDHWM, 2)
            self._publisher.setsockopt(zmq.LINGER, 0)
            self._publisher.bind(self.config.endpoint)
        self._thread = threading.Thread(target=self._run, name="local-camera-source", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        log.info("stopping local camera source endpoint=%s", self.config.endpoint)
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)
        self._close_camera()
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

        try:
            self._open_camera(cv2)
            log.info("local camera opened backend=%s endpoint=%s", self._active_backend, self.config.endpoint)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
            interval_s = max(self.config.read_interval_ms, 1) / 1000.0
            while not self._stop_event.is_set():
                ok, frame = self._read_frame(cv2)
                if not ok or frame is None:
                    self._record_error(f"camera read failed backend={self._active_backend or self.config.backend}")
                    time.sleep(interval_s)
                    continue

                ok, encoded = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    self._record_error("failed to encode camera frame as JPEG")
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
                    if self._last_error is not None:
                        log.info("local camera recovered after error: %s", self._last_error)
                        self._last_error = None

                if self._publisher is not None:
                    self._publisher.send(payload)
                time.sleep(interval_s)
        except Exception as exc:
            self._record_error(str(exc), exc_info=True)
        finally:
            log.info("local camera source stopped backend=%s frames=%s", self._active_backend, self._frames_received)
            self._close_camera()

    def _open_camera(self, cv2) -> None:
        backend = self.config.backend.strip().lower()
        if backend == "openni":
            self._open_openni_camera()
            self._active_backend = "openni"
            return
        self._capture = self._open_cv_capture(cv2)
        self._active_backend = backend

    def _open_openni_camera(self) -> None:
        try:
            from openni import openni2
        except ImportError as exc:
            raise RuntimeError("python package 'openni' is not installed in the container") from exc

        self._openni2 = openni2
        init_errors: list[str] = []
        log.info(
            "opening Astra S through OpenNI candidates=%s",
            ", ".join(str(value) for value in _openni_redist_candidates()),
        )
        for redist in _openni_redist_candidates():
            try:
                self._openni2.initialize(redist)
                self._oni_device = self._openni2.Device.open_any()
                log.info("OpenNI initialized redist=%s", redist or "<default>")
                break
            except Exception as exc:
                init_errors.append(f"{redist or '<default>'}: {type(exc).__name__}: {exc}")
                try:
                    self._openni2.unload()
                except Exception:
                    pass
                self._oni_device = None
        if self._oni_device is None:
            joined = "; ".join(init_errors) if init_errors else "no OpenNI2 redist candidates found"
            raise RuntimeError(f"failed to open OpenNI device: {joined}")
        self._oni_depth_stream = self._oni_device.create_depth_stream()
        self._oni_depth_stream.set_video_mode(
            self._openni2.c_api.OniVideoMode(
                pixelFormat=self._openni2.PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=self.config.width,
                resolutionY=self.config.height,
                fps=self.config.fps,
            )
        )
        self._oni_depth_stream.start()
        self._oni_color_stream = self._oni_device.create_color_stream()
        self._oni_color_stream.set_video_mode(
            self._openni2.c_api.OniVideoMode(
                pixelFormat=self._openni2.PIXEL_FORMAT_RGB888,
                resolutionX=self.config.width,
                resolutionY=self.config.height,
                fps=self.config.fps,
            )
        )
        self._oni_color_stream.start()
        try:
            self._oni_device.set_image_registration_mode(self._openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        except Exception:
            pass

    def _open_cv_capture(self, cv2):
        backend = self.config.backend.strip().lower()
        api_preference = getattr(cv2, "CAP_V4L2", None) if backend == "v4l2" else None
        if api_preference is None:
            capture = cv2.VideoCapture(self.config.rgb_device or "/dev/video0")
        else:
            capture = cv2.VideoCapture(self.config.rgb_device or "/dev/video0", api_preference)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open local camera backend={self.config.backend}")
        return capture

    def _read_frame(self, cv2):
        if self._active_backend == "openni":
            if self._oni_color_stream is None:
                return False, None
            try:
                color_frame = self._oni_color_stream.read_frame()
                rgb_buf = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                frame = rgb_buf.reshape(self.config.height, self.config.width, 3)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, bgr
            except Exception:
                return False, None
        if self._capture is None:
            return False, None
        return self._capture.read()

    def _close_camera(self) -> None:
        if self._oni_color_stream is not None:
            self._oni_color_stream.stop()
            self._oni_color_stream = None
        if self._oni_depth_stream is not None:
            self._oni_depth_stream.stop()
            self._oni_depth_stream = None
        if self._oni_device is not None:
            self._oni_device.close()
            self._oni_device = None
        if self._openni2 is not None:
            try:
                self._openni2.unload()
            except Exception:
                pass
            self._openni2 = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._active_backend = None

    def _record_error(self, message: str, *, exc_info: bool = False) -> None:
        with self._lock:
            self._last_error = message
        now = time.monotonic()
        if now - self._last_error_logged_monotonic < self.config.error_log_interval_s:
            return
        self._last_error_logged_monotonic = now
        log.error("local camera error: %s", message, exc_info=exc_info)


def _looks_like_jpeg(payload: bytes) -> bool:
    return len(payload) > 4 and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()


def _openni_redist_candidates() -> list[str | None]:
    candidates: list[str | None] = []
    for value in (
        os.environ.get("OPENNI2_REDIST"),
        os.path.join(os.environ.get("OPENNI_SDK_ROOT", "/opt/orbbec/openni"), "Redist"),
        "/usr/lib",
        None,
    ):
        if value is None:
            candidates.append(None)
            continue
        if _path_exists(value) and value not in candidates:
            candidates.append(value)
    return candidates
