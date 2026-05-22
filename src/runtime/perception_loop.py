from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from src.perception.sensor_fusion import FusedEntity
from src.runtime.frame_source import CameraFrame
from src.runtime.sensor_store import SensorStore
from src.transport.results import (
    PerceptionResultMessage,
    RuntimeMetricsMessage,
    TrackedEntityMessage,
)


class FrameSource(Protocol):
    frame_ready: threading.Event
    def latest(self) -> CameraFrame | None: ...


class PerceptionPipeline(Protocol):
    def process(
        self,
        frame_bgr: np.ndarray,
        depth_map_m: np.ndarray | None = None,
        lidar_scan: np.ndarray | None = None,
        accel_xyz_mps2: np.ndarray | None = None,
        quat_xyzw: np.ndarray | None = None,
        timestamp_us: int = 0,
        frame_id: int | None = None,
        delta_time_s: float = 0.1,
    ) -> tuple[list[FusedEntity], dict[str, float]]: ...


class ResultPublisher(Protocol):
    def publish(self, message: PerceptionResultMessage) -> None: ...


@dataclass(frozen=True, slots=True)
class PerceptionLoopConfig:
    source_id: str = "adaptive-runtime"
    interval_ms: int = 33  # ~30 FPS poll rate (was 100ms = max 10 FPS)


@dataclass(frozen=True, slots=True)
class PerceptionLoopStats:
    running: bool
    frames_processed: int
    publish_errors: int
    last_result_age_ms: float | None
    last_error: str | None


class RuntimePerceptionLoop:
    """Continuously turns live camera + ROS2 sensor state into result-plane messages."""

    def __init__(
        self,
        *,
        pipeline: PerceptionPipeline,
        sensor_store: SensorStore,
        frame_source: FrameSource,
        result_publisher: ResultPublisher,
        config: PerceptionLoopConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.sensor_store = sensor_store
        self.frame_source = frame_source
        self.result_publisher = result_publisher
        self.config = config or PerceptionLoopConfig()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frames_processed = 0
        self._publish_errors = 0
        self._last_result_monotonic: float | None = None
        self._last_error: str | None = None
        self._last_frame_sequence: int | None = None
        self._last_frame_timestamp_us: int | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="adaptive-perception-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        self.frame_source.frame_ready.set()
        if self._thread:
            self._thread.join(timeout=timeout_s)

    def stats(self) -> PerceptionLoopStats:
        with self._lock:
            age_ms = None
            if self._last_result_monotonic is not None:
                age_ms = (time.monotonic() - self._last_result_monotonic) * 1000.0
            return PerceptionLoopStats(
                running=bool(self._thread and self._thread.is_alive()),
                frames_processed=self._frames_processed,
                publish_errors=self._publish_errors,
                last_result_age_ms=age_ms,
                last_error=self._last_error,
            )

    def process_once(self, frame: CameraFrame) -> PerceptionResultMessage:
        decode_start = perf_counter()
        if frame.frame_bgr is not None:
            frame_bgr = frame.frame_bgr
        else:
            frame_bgr = decode_jpeg_frame(frame.payload)
        camera_ms = (perf_counter() - decode_start) * 1000.0
        delta_time_s = self._delta_time_s(frame)

        lidar = self.sensor_store.lidar_nearest(frame.timestamp_us) or self.sensor_store.latest_lidar()
        imu = self.sensor_store.imu_nearest(frame.timestamp_us) or self.sensor_store.latest_imu()
        entities, timings = self.pipeline.process(
            frame_bgr,
            depth_map_m=frame.depth_map_m,
            lidar_scan=lidar.scan_points if lidar is not None else None,
            accel_xyz_mps2=imu.accel_xyz_mps2 if imu is not None else None,
            quat_xyzw=imu.quat_xyzw if imu is not None else None,
            timestamp_us=frame.timestamp_us,
            frame_id=frame.sequence,
            delta_time_s=delta_time_s,
        )
        message = build_result_message(
            source_id=self.config.source_id,
            sequence=frame.sequence,
            timestamp_us=frame.timestamp_us,
            entities=entities,
            timings={**timings, "camera_ms": camera_ms},
        )
        self.result_publisher.publish(message)
        self._last_frame_sequence = frame.sequence
        self._last_frame_timestamp_us = frame.timestamp_us
        with self._lock:
            self._frames_processed += 1
            self._last_result_monotonic = time.monotonic()
            self._last_error = None
        return message

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.frame_source.frame_ready.wait()
            self.frame_source.frame_ready.clear()
            if self._stop_event.is_set():
                break
            frame = self.frame_source.latest()
            if frame is None or frame.sequence == self._last_frame_sequence:
                continue
            try:
                self.process_once(frame)
            except Exception as exc:
                with self._lock:
                    self._publish_errors += 1
                    self._last_error = str(exc)
                # Short sleep on error to avoid tight spin — do NOT sleep interval_s
                time.sleep(0.005)

    def _delta_time_s(self, frame: CameraFrame) -> float:
        if self._last_frame_timestamp_us is None:
            return max(self.config.interval_ms / 1000.0, 1e-3)
        delta_us = frame.timestamp_us - self._last_frame_timestamp_us
        return max(delta_us / 1_000_000.0, 1e-3)


def decode_jpeg_frame(payload: bytes) -> np.ndarray:
    import cv2

    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("cannot decode SCADA camera JPEG frame")
    return frame


def build_result_message(
    *,
    source_id: str,
    sequence: int,
    timestamp_us: int,
    entities: list[FusedEntity],
    timings: dict[str, float],
) -> PerceptionResultMessage:
    total_ms = float(timings.get("total_ms", 0.0))
    if total_ms <= 0.0:
        total_ms = sum(float(value) for value in timings.values())
    # FPS reflects pipeline throughput — exclude camera_ms (I/O, not compute)
    pipeline_ms = total_ms - float(timings.get("camera_ms", 0.0))
    fps = 1000.0 / pipeline_ms if pipeline_ms > 0.0 else 0.0
    return PerceptionResultMessage(
        source_id=source_id,
        sequence=sequence,
        timestamp_us=timestamp_us,
        entities=[_to_tracked_entity(entity) for entity in entities],
        metrics=RuntimeMetricsMessage(
            total_latency_ms=total_ms,
            camera_latency_ms=float(timings.get("camera_ms", 0.0)),
            detector_latency_ms=float(timings.get("detector_ms", 0.0)),
            fusion_latency_ms=float(timings.get("fusion_ms", 0.0)),
            fps=fps,
        ),
    )


def _to_tracked_entity(entity: FusedEntity) -> TrackedEntityMessage:
    return TrackedEntityMessage(
        track_id=int(entity.track_id),
        class_id=float(entity.class_id),
        bbox_xywh=entity.bbox_xywh.astype(np.float32, copy=True),
        contour_xy=entity.contour_xy.astype(np.float32, copy=True),
        contour_points_xyz_m=entity.contour_points_xyz_m.astype(np.float32, copy=True),
        position_xyz_m=entity.position_3d.astype(np.float32, copy=True),
        velocity_xyz_mps=entity.velocity_3d.astype(np.float32, copy=True),
        heading_rad=float(entity.heading_rad),
        confidence=float(entity.confidence),
        nearest_obstacle_distance_m=entity.nearest_obstacle_distance_m,
        distance_to_robot_m=entity.distance_to_robot_m,
        distance_source=entity.distance_source,
        sync_confidence=float(entity.sync_confidence),
    )
