from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imu_fusion import EgoMotionState
from .lidar_proc import LidarCluster
from .tracker import TrackState


@dataclass(slots=True)
class FusedEntity:
    track_id: int
    bbox_xywh: np.ndarray
    position_3d: np.ndarray
    velocity_3d: np.ndarray
    heading_rad: float
    confidence: float
    nearest_obstacle_distance_m: float | None
    nearest_obstacle_centroid_xy: np.ndarray | None
    ego_velocity_xyz_mps: np.ndarray


class SensorFusion:
    def fuse(
        self,
        tracks: list[TrackState],
        ego_motion: EgoMotionState,
        lidar_clusters: list[LidarCluster],
    ) -> list[FusedEntity]:
        fused: list[FusedEntity] = []
        for track in tracks:
            nearest_distance, nearest_centroid = self._nearest_obstacle(track.position_3d, lidar_clusters)
            fused.append(
                FusedEntity(
                    track_id=track.track_id,
                    bbox_xywh=track.bbox_xywh.copy(),
                    position_3d=track.position_3d.copy(),
                    velocity_3d=track.velocity_3d.copy(),
                    heading_rad=ego_motion.heading_rad,
                    confidence=track.confidence,
                    nearest_obstacle_distance_m=nearest_distance,
                    nearest_obstacle_centroid_xy=nearest_centroid,
                    ego_velocity_xyz_mps=ego_motion.velocity_xyz_mps.copy(),
                )
            )
        return fused

    @staticmethod
    def _nearest_obstacle(
        position_3d: np.ndarray,
        lidar_clusters: list[LidarCluster],
    ) -> tuple[float | None, np.ndarray | None]:
        if not lidar_clusters:
            return None, None

        target_xy = position_3d[:2]
        distances = [float(np.linalg.norm(cluster.centroid_xy - target_xy)) for cluster in lidar_clusters]
        nearest_index = int(np.argmin(distances))
        nearest_cluster = lidar_clusters[nearest_index]
        return distances[nearest_index], nearest_cluster.centroid_xy.copy()
