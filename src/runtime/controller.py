from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 on Jetson
    from strenum import StrEnum

from src.comm.health_monitor import HeartbeatClientDaemon
from src.runtime.camera import AstraSCameraConfig, AstraSCameraRuntime
from src.runtime.frame_source import (
    LocalCameraFrameConfig,
    LocalCameraFrameSource,
    ZmqJpegFrameConfig,
    ZmqJpegFrameReceiver,
)
from src.runtime.perception_loop import PerceptionLoopConfig, RuntimePerceptionLoop
from src.runtime.sensor_store import SensorStore
from src.transport.zmq_result_publisher import ZmqResultPublisher, ZmqResultPublisherConfig
from src.transport.zmq_sensor_ingest import SensorIngestStats, ZmqIngestConfig, ZmqSensorIngest
from src.utils.constants import FRAME_HEIGHT, FRAME_WIDTH
from src.utils.enums import SafetyState

log = logging.getLogger(__name__)


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    bind_host: str | None = None
    jetson_host: str = "25.12.4.100"
    pi_host: str = "25.12.4.101"
    runtime_backend: str = "engine"
    perception_enabled: bool = True
    perception_interval_ms: int = 33  # ~30 FPS (was 100ms = max 10 FPS)
    result_source_id: str = "adaptive-runtime"
    sensor_ingest_port: int = 5555
    result_publish_port: int = 5556
    heartbeat_port: int = 9093
    heartbeat_interval_ms: int = 500
    max_sensor_age_ms: int = 250
    engine_path: str = "/app/models/engines/best.engine"
    pt_model_path: str = "/app/models/fine_tuning/best.pt"
    camera_source: str = "device"
    scada_camera_host: str = "25.12.4.101"
    scada_camera_port: int = 5557
    camera_frame_timeout_ms: int = 2000
    camera_rgb_device: str = "/dev/video0"
    camera_depth_device: str = "/dev/video1"
    camera_width: int = FRAME_WIDTH
    camera_height: int = FRAME_HEIGHT
    camera_fps: int = 30
    camera_publish_port: int = 5557
    detect_every_n: int = 3
    """Run full detector every N frames; tracker propagates bboxes on skip frames.
    Default 3 reduces avg detector latency by ~3x (e.g. 85ms → ~29ms avg).
    Set to 1 to detect every frame (maximum accuracy, highest latency).
    """
    detector_input_scale: float = 1.0
    """Downscale factor for detector input frame (e.g. 0.5 = 320x240 from 640x480).
    Set to 0.5 for additional ~2-4x inference speedup on GPU/TRT.
    """


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    state: RuntimeState
    ready: bool
    reason: str | None
    ingest: SensorIngestStats
    engine_available: bool
    camera_available: bool
    perception_running: bool
    frames_processed: int
    last_result_age_ms: float | None
    last_runtime_error: str | None
    result_endpoint: str
    heartbeat_endpoint: str


class JetsonRuntimeController:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        sensor_store: SensorStore | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.sensor_store = sensor_store or SensorStore()
        self._state = RuntimeState.STOPPED
        self._reason: str | None = None
        self._camera = AstraSCameraRuntime(
            AstraSCameraConfig(
                backend=os.environ.get("CTX_CAMERA_BACKEND", "openni"),
                rgb_device=self.config.camera_rgb_device,
                depth_device=self.config.camera_depth_device,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
            )
        )
        self._ingest = ZmqSensorIngest(
            ZmqIngestConfig(
                bind_host=self._bind_host(),
                bind_port=self.config.sensor_ingest_port,
            ),
            handler=self.sensor_store.update,
        )
        self._result_publisher = ZmqResultPublisher(
            ZmqResultPublisherConfig(
                bind_host=self._bind_host(),
                bind_port=self.config.result_publish_port,
            )
        )
        self._frame_source: LocalCameraFrameSource | ZmqJpegFrameReceiver | None = self._make_frame_source()
        self._perception_loop: RuntimePerceptionLoop | None = None
        self._heartbeat_client: HeartbeatClientDaemon | None = None
        self._last_logged_status_key: tuple[object, ...] | None = None

    def start(self) -> RuntimeStatus:
        if self._state is RuntimeState.RUNNING:
            return self.status()
        self._state = RuntimeState.STARTING
        log.info(
            "runtime starting backend=%s camera_source=%s camera_backend=%s engine=%s pt=%s",
            self.config.runtime_backend,
            self.config.camera_source,
            os.environ.get("CTX_CAMERA_BACKEND", "openni"),
            self.config.engine_path,
            self.config.pt_model_path,
        )
        try:
            log.info("starting sensor ingest endpoint=tcp://%s:%s", self._bind_host(), self.config.sensor_ingest_port)
            self._ingest.start()
            log.info("starting result publisher endpoint=%s", self._result_publisher.config.endpoint)
            self._result_publisher.start()
            self._start_frame_source()
            self._start_perception_loop()
            self._start_heartbeat()
        except Exception as exc:
            log.exception("runtime startup failed")
            self._stop_perception_loop()
            self._stop_frame_source()
            self._ingest.stop()
            self._result_publisher.stop()
            self._stop_heartbeat()
            self._state = RuntimeState.ERROR
            self._reason = str(exc)
            return self.status()
        self._state = RuntimeState.RUNNING
        self._reason = None
        status = self.status()
        log.info(
            "runtime started ready=%s reason=%s result_endpoint=%s heartbeat_endpoint=%s",
            status.ready,
            status.reason,
            status.result_endpoint,
            status.heartbeat_endpoint,
        )
        return status

    def stop(self) -> RuntimeStatus:
        log.info("runtime stopping")
        self._stop_perception_loop()
        self._stop_frame_source()
        self._stop_heartbeat()
        self._ingest.stop()
        self._result_publisher.stop()
        self._state = RuntimeState.STOPPED
        self._reason = None
        return self.status()

    def status(self) -> RuntimeStatus:
        snapshot = self.sensor_store.snapshot()
        ingest_stats = self._ingest.stats()
        perception_stats = self._perception_loop.stats() if self._perception_loop is not None else None
        reason = self._reason
        inference_artifact_available = self._inference_artifact_available()
        camera_available = self._camera_available()
        ready = (
            self._state is RuntimeState.RUNNING
            and inference_artifact_available
            and camera_available
            and snapshot.has_lidar
            and snapshot.has_imu
        )
        if self._state is RuntimeState.RUNNING:
            if not inference_artifact_available:
                reason = "waiting for inference engine"
            elif not camera_available:
                reason = self._camera_wait_reason()
            elif not snapshot.has_lidar:
                reason = "waiting for lidar stream from raspberry pi"
            elif not snapshot.has_imu:
                reason = "waiting for imu stream from raspberry pi"
            elif _is_stale(snapshot.last_lidar_age_ms, self.config.max_sensor_age_ms):
                ready = False
                reason = "lidar stream is stale"
            elif _is_stale(snapshot.last_imu_age_ms, self.config.max_sensor_age_ms):
                ready = False
                reason = "imu stream is stale"
            else:
                reason = None
        status = RuntimeStatus(
            state=self._state,
            ready=ready,
            reason=reason,
            ingest=ingest_stats,
            engine_available=inference_artifact_available,
            camera_available=camera_available,
            perception_running=bool(perception_stats.running if perception_stats is not None else False),
            frames_processed=perception_stats.frames_processed if perception_stats is not None else 0,
            last_result_age_ms=perception_stats.last_result_age_ms if perception_stats is not None else None,
            last_runtime_error=perception_stats.last_error if perception_stats is not None else None,
            result_endpoint=self._result_publisher.config.endpoint,
            heartbeat_endpoint=f"tcp://{self.config.pi_host}:{self.config.heartbeat_port}",
        )
        self._log_status_change(status)
        return status

    def _camera_available(self) -> bool:
        if self.config.camera_source == "scada_zmq":
            if self._frame_source is None:
                return False
            stats = self._frame_source.stats()
            return (
                stats.running
                and stats.last_frame_age_ms is not None
                and stats.last_frame_age_ms <= self.config.camera_frame_timeout_ms
            )
        if self.config.camera_source == "device" and self._frame_source is not None:
            if self._state is RuntimeState.STOPPED:
                try:
                    self._camera.assert_available()
                except Exception:
                    return False
                return True
            stats = self._frame_source.stats()
            return (
                stats.running
                and stats.last_frame_age_ms is not None
                and stats.last_frame_age_ms <= self.config.camera_frame_timeout_ms
            )
        try:
            self._camera.assert_available()
        except Exception:
            return False
        return True

    def _camera_wait_reason(self) -> str:
        if self.config.camera_source == "scada_zmq":
            return "waiting for Wheeltec SCADA camera frames"
        if self.config.camera_source == "device":
            if self._frame_source is not None:
                stats = self._frame_source.stats()
                if stats.last_error:
                    return stats.last_error
            return "waiting for Astra S OpenNI camera frames"
        return "waiting for Astra S OpenNI camera device"

    def _inference_artifact_available(self) -> bool:
        if self.config.runtime_backend == "pt":
            return Path(self.config.pt_model_path).exists()
        return Path(self.config.engine_path).exists()

    def _bind_host(self) -> str:
        return self.config.bind_host or self.config.jetson_host

    def _make_frame_source(self) -> LocalCameraFrameSource | ZmqJpegFrameReceiver | None:
        if self.config.camera_source == "scada_zmq":
            return ZmqJpegFrameReceiver(
                ZmqJpegFrameConfig(
                    host=self.config.scada_camera_host,
                    port=self.config.scada_camera_port,
                )
            )
        if self.config.camera_source == "device":
            return LocalCameraFrameSource(
                LocalCameraFrameConfig(
                    backend=os.environ.get("CTX_CAMERA_BACKEND", "openni"),
                    rgb_device=self.config.camera_rgb_device,
                    width=self.config.camera_width,
                    height=self.config.camera_height,
                    fps=self.config.camera_fps,
                    read_interval_ms=max(1, int(1000 / max(self.config.camera_fps, 1))),
                    publish_port=self.config.camera_publish_port,
                    publish_enabled=_env_bool("CTX_CAMERA_PUBLISH_ENABLED", True),
                )
            )
        return None

    def _start_frame_source(self) -> None:
        if self._frame_source is not None:
            log.info("starting frame source endpoint=%s", self._frame_source.config.endpoint)
            self._frame_source.start()

    def _stop_frame_source(self) -> None:
        if self._frame_source is not None:
            log.info("stopping frame source endpoint=%s", self._frame_source.config.endpoint)
            self._frame_source.stop()

    def _start_perception_loop(self) -> None:
        if not self.config.perception_enabled or self._perception_loop is not None:
            if not self.config.perception_enabled:
                log.warning("perception loop disabled by CTX_PERCEPTION_ENABLED")
            return
        if self._frame_source is None:
            log.warning("perception loop not started because frame source is not configured")
            return

        from src.perception.pipeline import PerceptionPipeline

        log.info("starting perception loop interval_ms=%s", self.config.perception_interval_ms)
        self._perception_loop = RuntimePerceptionLoop(
            pipeline=PerceptionPipeline(
                detector=_make_detector(self.config),
                detect_every_n=self.config.detect_every_n,
            ),
            sensor_store=self.sensor_store,
            frame_source=self._frame_source,
            result_publisher=self._result_publisher,
            config=PerceptionLoopConfig(
                source_id=self.config.result_source_id,
                interval_ms=self.config.perception_interval_ms,
            ),
        )
        self._perception_loop.start()

    def _stop_perception_loop(self) -> None:
        if self._perception_loop is None:
            return
        log.info("stopping perception loop")
        self._perception_loop.stop()
        self._perception_loop = None

    def _start_heartbeat(self) -> None:
        if self._heartbeat_client is not None:
            return
        log.info("starting heartbeat client endpoint=tcp://%s:%s", self.config.pi_host, self.config.heartbeat_port)
        self._heartbeat_client = HeartbeatClientDaemon(
            host=self.config.pi_host,
            port=self.config.heartbeat_port,
            timeout_s=0.5,
            interval_s=self.config.heartbeat_interval_ms / 1000.0,
            heartbeat_payload_factory=self._heartbeat_payload,
        )
        self._heartbeat_client.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_client is None:
            return
        log.info("stopping heartbeat client endpoint=tcp://%s:%s", self.config.pi_host, self.config.heartbeat_port)
        self._heartbeat_client.stop()
        self._heartbeat_client = None

    def _heartbeat_payload(self) -> dict[str, float | int]:
        status = self.status()
        state = SafetyState.NORMAL if status.ready else SafetyState.DEGRADED
        return {
            "state": int(state),
            "pipeline_fps": 0.0,
            "gpu_temp_c": 0,
        }

    def _log_status_change(self, status: RuntimeStatus) -> None:
        key = (
            status.state,
            status.ready,
            status.reason,
            status.engine_available,
            status.camera_available,
            status.ingest.running,
            status.perception_running,
        )
        if key == self._last_logged_status_key:
            return
        self._last_logged_status_key = key
        message = (
            "runtime status state=%s ready=%s reason=%s engine_available=%s camera_available=%s "
            "ingest_running=%s perception_running=%s frames_processed=%s messages_received=%s"
        )
        args = (
            status.state.value,
            status.ready,
            status.reason,
            status.engine_available,
            status.camera_available,
            status.ingest.running,
            status.perception_running,
            status.frames_processed,
            status.ingest.messages_received,
        )
        if status.ready:
            log.info(message, *args)
        else:
            log.warning(message, *args)


def _is_stale(age_ms: float | None, max_age_ms: int) -> bool:
    return age_ms is None or age_ms > max_age_ms


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _make_detector(config: RuntimeConfig):
    """Build a :class:`PersonDetector` wired to the given runtime config."""
    from src.perception.detector import DetectorConfig, PersonDetector

    return PersonDetector(
        DetectorConfig(
            backend=config.runtime_backend,
            engine_path=Path(config.engine_path),
            pt_model_path=Path(config.pt_model_path),
            input_scale=config.detector_input_scale,
        )
    )
