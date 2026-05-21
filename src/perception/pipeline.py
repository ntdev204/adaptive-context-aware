"""pipeline.py — End-to-end perception pipeline.

Supports any :class:`InputSource` (image, video, or camera).
Depth map is **optional**; when omitted a flat zero-depth plane is used
so the tracker still runs in 2-D mode with position_3d.z = 0.

Example – run on a video file::

    pipeline = PerceptionPipeline()
    source   = InputSource.from_video("campus.mp4")

    with source:
        for entities, timings in pipeline.run_source(source):
            print(timings["total_ms"])

Example – run on a single image::

    pipeline = PerceptionPipeline()
    source   = InputSource.from_image("photo.jpg")
    with source:
        for entities, timings in pipeline.run_source(source):
            ...

Example – run on live camera::

    pipeline = PerceptionPipeline()
    source   = InputSource.from_camera(device_index=0)
    with source:
        for entities, timings in pipeline.run_source(source):
            ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Generator

import numpy as np

from src.utils.constants import DEPTH_SHAPE_HW

from .depth_proc import CameraIntrinsics, DepthProcessor
from .detector import DetectorConfig, PersonDetector
from .feature_extractor import EntityFeatureExtractor
from .imu_fusion import IMUFusion
from .input_source import InputSource
from .lidar_proc import LidarProcessor
from .sensor_fusion import FusedEntity, SensorFusion
from .tracker import MultiObjectTracker

_FLAT_DEPTH_Z_M = 5.0  # assumed depth when no real depth map is available
_CACHED_FLAT_DEPTH: np.ndarray | None = None


def _make_flat_depth() -> np.ndarray:
    """Return a cached constant-depth plane (all pixels = _FLAT_DEPTH_Z_M)."""
    global _CACHED_FLAT_DEPTH
    if _CACHED_FLAT_DEPTH is None:
        _CACHED_FLAT_DEPTH = np.full(DEPTH_SHAPE_HW, _FLAT_DEPTH_Z_M, dtype=np.float32)
        _CACHED_FLAT_DEPTH.flags.writeable = False
    return _CACHED_FLAT_DEPTH


@dataclass(slots=True)
class PerceptionPipelineReport:
    fps: float
    latency_ms: dict[str, float]
    peak_rss_mb: float
    entity_count: int


@dataclass(slots=True)
class PerceptionPipeline:
    detector: PersonDetector = field(
        default_factory=lambda: PersonDetector(DetectorConfig(backend=os.environ.get("CTX_RUNTIME_BACKEND", "engine")))
    )
    depth_processor: DepthProcessor = field(
        default_factory=lambda: DepthProcessor(CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0))
    )
    tracker: MultiObjectTracker = field(init=False)
    lidar_processor: LidarProcessor = field(default_factory=LidarProcessor)
    imu_fusion: IMUFusion = field(default_factory=IMUFusion)
    sensor_fusion: SensorFusion = field(default_factory=SensorFusion)
    feature_extractor: EntityFeatureExtractor = field(default_factory=EntityFeatureExtractor)
    tracker_config_path: str = field(
        default_factory=lambda: os.environ.get("CTX_TRACKER_CONFIG_PATH", "models/fine_tuning/botsort_tuned.json")
    )

    def __post_init__(self) -> None:
        self.tracker = self._load_tracker(self.tracker_config_path)

    def warmup(self) -> None:
        self.detector.warmup()

    def close(self) -> None:
        self.detector.close()

    # ------------------------------------------------------------------
    # Tracker factory
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tracker(config_path: str) -> MultiObjectTracker:
        """Load best BoT-SORT config from JSON, fall back to defaults."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cfg = data.get("config", {})
                return MultiObjectTracker(
                    iou_threshold=cfg.get("iou_threshold", 0.3),
                    depth_gate_m=cfg.get("depth_gate_m", 1.0),
                    max_missed_frames=cfg.get("max_missed_frames", 3),
                )
        except (FileNotFoundError, json.JSONDecodeError):
            return MultiObjectTracker()

    # ------------------------------------------------------------------
    # Core frame processor
    # ------------------------------------------------------------------

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
    ) -> tuple[list[FusedEntity], dict[str, float]]:
        """Process a single BGR frame.

        Args:
            frame_bgr:    480×640 uint8 BGR frame (required).
            depth_map_m:  480×640 float32 depth map in metres. Pass ``None``
                          to use a flat virtual depth (5 m, 2-D tracking only).
            timestamp_us: Monotonic timestamp in microseconds.
            frame_id:     Optional frame index (used by synthetic backend).
            delta_time_s: Elapsed time since the previous frame (for velocity).

        Returns:
            (entities, timings) where timings maps stage name → ms elapsed.
        """
        timings: dict[str, float] = {}

        if depth_map_m is None:
            depth_map_m = _make_flat_depth()

        start = perf_counter()
        detections = self.detector.detect(frame_bgr, frame_id=frame_id).detections
        timings["detector_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        depth_boxes = self.depth_processor.detections_to_3d(depth_map_m, detections)
        timings["depth_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        tracks = self.tracker.update(detections, depth_boxes, delta_time_s=delta_time_s)
        timings["tracker_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        ego_motion = None
        if accel_xyz_mps2 is not None and quat_xyzw is not None:
            ego_motion = self.imu_fusion.update(accel_xyz_mps2, quat_xyzw, timestamp_us)
        lidar_clusters = self.lidar_processor.cluster_scan(lidar_scan) if lidar_scan is not None else []
        fused = self.sensor_fusion.fuse(tracks, ego_motion, lidar_clusters)
        timings["fusion_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        _features = [self.feature_extractor.extract(entity) for entity in fused]
        timings["feature_ms"] = (perf_counter() - start) * 1000.0

        timings["total_ms"] = sum(timings.values())
        return fused, timings

    # ------------------------------------------------------------------
    # Convenience: iterate over any InputSource
    # ------------------------------------------------------------------

    def run_source(
        self,
        source: InputSource,
        depth_map_m: np.ndarray | None = None,
    ) -> Generator[tuple[list[FusedEntity], dict[str, float]], None, None]:
        """Iterate over every frame from *source* and yield pipeline results.

        The source must already be opened (e.g. via a ``with`` block).

        Args:
            source:      An opened :class:`InputSource` instance.
            depth_map_m: Optional constant depth map to use for every frame.
                         Pass ``None`` for flat-depth mode.

        Yields:
            (entities, timings) for each frame.
        """
        prev_ts_us: int | None = None
        for frame_bgr, frame_id, timestamp_us in source:
            delta = (timestamp_us - prev_ts_us) / 1_000_000.0 if prev_ts_us is not None else 0.1
            prev_ts_us = timestamp_us
            yield self.process(
                frame_bgr,
                depth_map_m=depth_map_m,
                lidar_scan=None,
                accel_xyz_mps2=None,
                quat_xyzw=None,
                timestamp_us=timestamp_us,
                frame_id=frame_id,
                delta_time_s=max(delta, 1e-6),
            )
