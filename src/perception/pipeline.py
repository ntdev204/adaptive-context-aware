from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .depth_proc import CameraIntrinsics, DepthProcessor
from .detector import DetectorConfig, PersonDetector
from .feature_extractor import EntityFeatureExtractor
from .imu_fusion import IMUFusion
from .lidar_proc import LidarProcessor
from .sensor_fusion import SensorFusion
from .tracker import MultiObjectTracker


@dataclass(slots=True)
class PerceptionPipelineReport:
    fps: float
    latency_ms: dict[str, float]
    peak_rss_mb: float
    entity_count: int


@dataclass(slots=True)
class PerceptionPipeline:
    detector: PersonDetector = field(
        default_factory=lambda: PersonDetector(DetectorConfig(backend="engine"))
    )
    depth_processor: DepthProcessor = field(
        default_factory=lambda: DepthProcessor(
            CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
        )
    )
    lidar_processor: LidarProcessor = field(default_factory=LidarProcessor)
    tracker: MultiObjectTracker = field(default_factory=MultiObjectTracker)
    imu_fusion: IMUFusion = field(default_factory=IMUFusion)
    sensor_fusion: SensorFusion = field(default_factory=SensorFusion)
    feature_extractor: EntityFeatureExtractor = field(default_factory=EntityFeatureExtractor)

    def process(
        self,
        frame_bgr: np.ndarray,
        depth_map_m: np.ndarray,
        lidar_scan: np.ndarray,
        imu_accel_xyz_mps2: np.ndarray,
        imu_quat_xyzw: np.ndarray,
        timestamp_us: int,
        frame_id: int | None = None,
    ) -> tuple[list[object], dict[str, float]]:
        timings: dict[str, float] = {}

        start = perf_counter()
        detections = self.detector.detect(frame_bgr, frame_id=frame_id).detections
        timings["detector_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        depth_boxes = self.depth_processor.detections_to_3d(depth_map_m, detections)
        timings["depth_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        lidar_clusters = self.lidar_processor.cluster_scan(lidar_scan)
        timings["lidar_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        tracks = self.tracker.update(detections, depth_boxes, delta_time_s=0.1)
        timings["tracker_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        ego_motion = self.imu_fusion.update(imu_accel_xyz_mps2, imu_quat_xyzw, timestamp_us=timestamp_us)
        timings["imu_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        fused = self.sensor_fusion.fuse(tracks, ego_motion, lidar_clusters)
        timings["fusion_ms"] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        _features = [self.feature_extractor.extract(entity) for entity in fused]
        timings["feature_ms"] = (perf_counter() - start) * 1000.0

        timings["total_ms"] = sum(timings.values())
        return fused, timings
