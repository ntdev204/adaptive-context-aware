from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 on Jetson
    from strenum import StrEnum

from src.comm.health_monitor import HeartbeatClientDaemon
from src.runtime.camera import AstraSCameraConfig, AstraSCameraRuntime
from src.runtime.sensor_store import SensorStore
from src.transport.zmq_result_publisher import ZmqResultPublisher, ZmqResultPublisherConfig
from src.transport.zmq_sensor_ingest import SensorIngestStats, ZmqIngestConfig, ZmqSensorIngest
from src.utils.constants import FRAME_HEIGHT, FRAME_WIDTH
from src.utils.enums import SafetyState


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    jetson_host: str = "127.0.0.1"
    pi_host: str = "25.12.4.101"
    runtime_backend: str = "engine"
    sensor_ingest_port: int = 5555
    result_publish_port: int = 5556
    heartbeat_port: int = 9093
    heartbeat_interval_ms: int = 500
    max_sensor_age_ms: int = 250
    engine_path: str = "/app/models/engines/best.engine"
    pt_model_path: str = "/app/models/fine_tuning/best.pt"
    camera_rgb_device: str = "/dev/video0"
    camera_depth_device: str = "/dev/video1"
    camera_width: int = FRAME_WIDTH
    camera_height: int = FRAME_HEIGHT
    camera_fps: int = 30


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    state: RuntimeState
    ready: bool
    reason: str | None
    ingest: SensorIngestStats
    engine_available: bool
    camera_available: bool
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
                backend=os.environ.get("CTX_CAMERA_BACKEND", "v4l2"),
                rgb_device=self.config.camera_rgb_device,
                depth_device=self.config.camera_depth_device,
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
            )
        )
        self._ingest = ZmqSensorIngest(
            ZmqIngestConfig(
                bind_host=self.config.jetson_host,
                bind_port=self.config.sensor_ingest_port,
            ),
            handler=self.sensor_store.update,
        )
        self._result_publisher = ZmqResultPublisher(
            ZmqResultPublisherConfig(
                bind_host=self.config.jetson_host,
                bind_port=self.config.result_publish_port,
            )
        )
        self._heartbeat_client: HeartbeatClientDaemon | None = None

    def start(self) -> RuntimeStatus:
        if self._state is RuntimeState.RUNNING:
            return self.status()
        self._state = RuntimeState.STARTING
        try:
            self._ingest.start()
            self._result_publisher.start()
            self._start_heartbeat()
        except Exception as exc:
            self._state = RuntimeState.ERROR
            self._reason = str(exc)
            return self.status()
        self._state = RuntimeState.RUNNING
        self._reason = None
        return self.status()

    def stop(self) -> RuntimeStatus:
        self._stop_heartbeat()
        self._ingest.stop()
        self._result_publisher.stop()
        self._state = RuntimeState.STOPPED
        self._reason = None
        return self.status()

    def status(self) -> RuntimeStatus:
        snapshot = self.sensor_store.snapshot()
        ingest_stats = self._ingest.stats()
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
                reason = "waiting for AstraS RGB-D camera devices"
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
        return RuntimeStatus(
            state=self._state,
            ready=ready,
            reason=reason,
            ingest=ingest_stats,
            engine_available=inference_artifact_available,
            camera_available=camera_available,
            result_endpoint=self._result_publisher.config.endpoint,
            heartbeat_endpoint=f"tcp://{self.config.pi_host}:{self.config.heartbeat_port}",
        )

    def _camera_available(self) -> bool:
        try:
            self._camera.assert_available()
        except Exception:
            return False
        return True

    def _inference_artifact_available(self) -> bool:
        if self.config.runtime_backend == "pt":
            return Path(self.config.pt_model_path).exists()
        return Path(self.config.engine_path).exists()

    def _start_heartbeat(self) -> None:
        if self._heartbeat_client is not None:
            return
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


def _is_stale(age_ms: float | None, max_age_ms: int) -> bool:
    return age_ms is None or age_ms > max_age_ms
