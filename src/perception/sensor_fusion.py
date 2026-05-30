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
        ego_motion: EgoMotionState | None = None,
        lidar_clusters: list[LidarCluster] | None = None,
    ) -> list[FusedEntity]:
        ego_velocity = (
            ego_motion.velocity_xyz_mps.astype(np.float32, copy=True)
            if ego_motion is not None
            else np.zeros(3, dtype=np.float32)
        )
        heading_rad = float(ego_motion.heading_rad) if ego_motion is not None else 0.0
        clusters = lidar_clusters or []

        fused: list[FusedEntity] = []
        for track in tracks:
            # Compute once — avoids calling _nearest_cluster twice per entity
            nearest_dist, nearest_centroid = _nearest_cluster(track, clusters)
            fused.append(
                FusedEntity(
                    track_id=track.track_id,
                    bbox_xywh=track.bbox_xywh.copy(),
                    position_3d=track.position_3d.copy(),
                    velocity_3d=track.velocity_3d.copy(),
                    heading_rad=heading_rad,
                    confidence=track.confidence,
                    nearest_obstacle_distance_m=nearest_dist,
                    nearest_obstacle_centroid_xy=nearest_centroid,
                    ego_velocity_xyz_mps=ego_velocity.copy(),
                )
            )
        return fused


def _nearest_cluster(track: TrackState, lidar_clusters: list[LidarCluster]) -> tuple[float | None, np.ndarray | None]:
    if not lidar_clusters:
        return None, None

    track_xy = np.asarray(track.position_3d[:2], dtype=np.float32)
    nearest_distance: float | None = None
    nearest_centroid: np.ndarray | None = None
    for cluster in lidar_clusters:
        centroid = np.asarray(cluster.centroid_xy, dtype=np.float32)
        distance = float(np.linalg.norm(centroid - track_xy))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_centroid = centroid.copy()
    return nearest_distance, nearest_centroid
