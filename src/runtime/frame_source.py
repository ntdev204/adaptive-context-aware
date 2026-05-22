from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq

MAX_JPEG_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CameraFrame:
    payload: bytes
    sequence: int
    timestamp_us: int
    received_monotonic: float
    frame_bgr: np.ndarray | None = None
    depth_map_m: np.ndarray | None = None


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
    publish_depth_preview: bool = False
    jpeg_quality: int = 70

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
        self.frame_ready = threading.Event()

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
            if not _looks_like_jpeg(raw) or len(raw) > MAX_JPEG_BYTES:
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
            self.frame_ready.set()


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
        self.frame_ready = threading.Event()

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

    def _run(self, cv2=None) -> None:
        if cv2 is None:
            import cv2

        try:
            self._open_camera(cv2)
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
            interval_s = max(self.config.read_interval_ms, 1) / 1000.0
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                ok, frame_bgr, depth_map_m = self._read_frame(cv2)
                if not ok or frame_bgr is None:
                    with self._lock:
                        self._last_error = f"camera read failed backend={self.config.backend}"
                    time.sleep(interval_s)
                    continue

                payload = b""
                depth_payload = b""
                if self._publisher is not None:
                    ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_params)
                    if not ok:
                        with self._lock:
                            self._last_error = "failed to encode camera frame as JPEG"
                        time.sleep(interval_s)
                        continue
                    payload = encoded.tobytes()
                    if len(payload) > MAX_JPEG_BYTES:
                        with self._lock:
                            self._last_error = f"encoded JPEG exceeds {MAX_JPEG_BYTES} bytes"
                        payload = b""

                    if self.config.publish_depth_preview and depth_map_m is not None:
                        depth_payload = _encode_depth_preview_jpeg(depth_map_m, cv2, encode_params)
                        if depth_payload:
                            depth_payload = b"DEPTH:" + depth_payload

                    if payload:
                        payload = b"RAW:" + payload

                _make_readonly(frame_bgr)
                if depth_map_m is not None:
                    _make_readonly(depth_map_m)

                now = time.monotonic()
                with self._lock:
                    self._frames_received += 1
                    camera_frame = CameraFrame(
                        payload=payload,
                        sequence=self._frames_received,
                        timestamp_us=int(time.time() * 1_000_000),
                        received_monotonic=now,
                        frame_bgr=frame_bgr,
                        depth_map_m=depth_map_m,
                    )
                    self._latest = camera_frame
                    if payload or self._publisher is None:
                        self._last_error = None
                self.frame_ready.set()

                if self._publisher is not None and payload:
                    self._publisher.send(payload)
                if self._publisher is not None and depth_payload:
                    self._publisher.send(depth_payload)
                elapsed_s = time.monotonic() - loop_start
                remaining_s = interval_s - elapsed_s
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            self._close_camera()

    def _open_camera(self, cv2) -> None:
        if self.config.backend == "openni":
            self._open_openni_camera()
            return
        self._capture = self._open_cv_capture(cv2)

    def _open_openni_camera(self) -> None:
        try:
            from openni import openni2
        except ImportError as exc:
            raise RuntimeError("python package 'openni' is not installed in the container") from exc

        self._openni2 = openni2
        init_errors: list[str] = []
        for redist in _openni_redist_candidates():
            try:
                self._openni2.initialize(redist)
                self._oni_device = self._openni2.Device.open_any()
                break
            except Exception as exc:
                init_errors.append(f"{redist or '<default>'}: {type(exc).__name__}")
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
        capture = cv2.VideoCapture(self.config.rgb_device or "/dev/video0")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"failed to open local camera backend={self.config.backend}")
        return capture

    def _read_frame(self, cv2):
        if self.config.backend == "openni":
            if self._oni_color_stream is None:
                return False, None, None
            try:
                color_frame = self._oni_color_stream.read_frame()
                rgb_buf = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
                frame = rgb_buf.reshape(self.config.height, self.config.width, 3)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                depth_map_m = self._read_openni_depth()
                return True, bgr, depth_map_m
            except Exception:
                return False, None, None
        if self._capture is None:
            return False, None, None
        ok, bgr = self._capture.read()
        return ok, bgr, None

    def _read_openni_depth(self) -> np.ndarray | None:
        if self._oni_depth_stream is None:
            return None
        try:
            depth_frame = self._oni_depth_stream.read_frame()
            depth_buf = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
            depth_mm = depth_buf.reshape(self.config.height, self.config.width)
            return depth_mm.astype(np.float32) / 1000.0
        except Exception:
            return None

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


def _looks_like_jpeg(payload: bytes) -> bool:
    return len(payload) > 4 and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


def _encode_depth_preview_jpeg(depth_map_m: np.ndarray, cv2, encode_params: list[int]) -> bytes:
    valid = depth_map_m[np.isfinite(depth_map_m) & (depth_map_m > 0)]
    if valid.size == 0:
        return b""
    min_depth = float(np.min(valid))
    max_depth = float(np.max(valid))
    scale = max(max_depth - min_depth, 1e-3)
    normalized = np.clip((depth_map_m - min_depth) / scale, 0.0, 1.0)
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".jpg", colored, encode_params)
    if not ok:
        return b""
    payload = encoded.tobytes()
    if len(payload) > MAX_JPEG_BYTES:
        return b""
    return payload


def _make_readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


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
