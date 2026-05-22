from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imu_fusion import EgoMotionState
from .lidar_proc import LidarCluster
from .tracker import TrackState


@dataclass(slots=True)
class FusedEntity:
    track_id: int
    class_id: float
    bbox_xywh: np.ndarray
    contour_xy: np.ndarray
    contour_points_xyz_m: np.ndarray
    position_3d: np.ndarray
    velocity_3d: np.ndarray
    heading_rad: float
    confidence: float
    nearest_obstacle_distance_m: float | None
    nearest_obstacle_centroid_xy: np.ndarray | None
    ego_velocity_xyz_mps: np.ndarray
    distance_to_robot_m: float | None
    distance_source: str | None
    sync_confidence: float


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

        return [self._fuse_track(track, heading_rad, ego_velocity, clusters) for track in tracks]

    @staticmethod
    def _fuse_track(
        track: TrackState,
        heading_rad: float,
        ego_velocity: np.ndarray,
        clusters: list[LidarCluster],
    ) -> FusedEntity:
        nearest_dist, nearest_centroid = _nearest_cluster(track, clusters)
        depth_distance = _depth_distance(track)
        if depth_distance is not None and nearest_dist is not None:
            distance_to_robot_m = min(depth_distance, nearest_dist)
            distance_source = "depth_lidar_fused"
            sync_confidence = 0.9
        elif depth_distance is not None:
            distance_to_robot_m = depth_distance
            distance_source = "depth_only"
            sync_confidence = 0.7
        elif nearest_dist is not None:
            distance_to_robot_m = nearest_dist
            distance_source = "lidar_only"
            sync_confidence = 0.5
        else:
            distance_to_robot_m = None
            distance_source = None
            sync_confidence = 0.0

        return FusedEntity(
            track_id=track.track_id,
            class_id=track.class_id,
            bbox_xywh=track.bbox_xywh.copy(),
            contour_xy=track.contour_xy.copy(),
            contour_points_xyz_m=track.contour_points_xyz_m.copy(),
            position_3d=track.position_3d.copy(),
            velocity_3d=track.velocity_3d.copy(),
            heading_rad=heading_rad,
            confidence=track.confidence,
            nearest_obstacle_distance_m=nearest_dist,
            nearest_obstacle_centroid_xy=nearest_centroid,
            ego_velocity_xyz_mps=ego_velocity.copy(),
            distance_to_robot_m=distance_to_robot_m,
            distance_source=distance_source,
            sync_confidence=sync_confidence,
        )


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


def _depth_distance(track: TrackState) -> float | None:
    if track.contour_points_xyz_m.size == 0:
        position = np.asarray(track.position_3d, dtype=np.float32)
        if position.shape[0] < 3:
            return None
        return float(np.linalg.norm(position))

    distances = np.linalg.norm(track.contour_points_xyz_m[:, :3], axis=1)
    if distances.size == 0:
        return None
    return float(np.min(distances))
