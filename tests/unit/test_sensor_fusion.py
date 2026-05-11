from __future__ import annotations

import numpy as np

from src.perception.imu_fusion import EgoMotionState
from src.perception.lidar_proc import LidarCluster
from src.perception.sensor_fusion import SensorFusion
from src.perception.tracker import TrackState


def _track(track_id: int, x: float, y: float, z: float) -> TrackState:
    return TrackState(
        track_id=track_id,
        bbox_xywh=np.array([100.0, 80.0, 40.0, 120.0], dtype=np.float32),
        position_3d=np.array([x, y, z], dtype=np.float32),
        velocity_3d=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        age=2,
        missed_frames=0,
        confidence=0.9,
    )


def _cluster(x: float, y: float) -> LidarCluster:
    points = np.array([[x, y], [x + 0.1, y + 0.1], [x - 0.1, y]], dtype=np.float32)
    return LidarCluster(
        points_xy=points,
        centroid_xy=np.array([x, y], dtype=np.float32),
        mean_range_m=float(np.mean(np.linalg.norm(points, axis=1))),
        radius_m=0.15,
    )


def test_sensor_fusion_outputs_complete_entity_fields() -> None:
    fusion = SensorFusion()
    tracks = [_track(track_id=1, x=1.0, y=0.5, z=2.0)]
    ego_motion = EgoMotionState(
        velocity_xyz_mps=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        heading_rad=0.3,
        timestamp_us=2_000_000,
    )
    lidar_clusters = [_cluster(1.2, 0.6)]

    fused = fusion.fuse(tracks, ego_motion, lidar_clusters)

    assert len(fused) == 1
    entity = fused[0]
    assert entity.track_id == 1
    assert entity.position_3d.shape == (3,)
    assert entity.velocity_3d.shape == (3,)
    assert entity.heading_rad == 0.3
    assert entity.nearest_obstacle_distance_m is not None
    assert entity.nearest_obstacle_centroid_xy is not None
    assert entity.ego_velocity_xyz_mps.shape == (3,)


def test_sensor_fusion_handles_missing_lidar_clusters() -> None:
    fusion = SensorFusion()
    tracks = [_track(track_id=1, x=0.0, y=0.0, z=2.0)]
    ego_motion = EgoMotionState(
        velocity_xyz_mps=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        heading_rad=0.0,
        timestamp_us=1_000_000,
    )

    fused = fusion.fuse(tracks, ego_motion, [])

    assert fused[0].nearest_obstacle_distance_m is None
    assert fused[0].nearest_obstacle_centroid_xy is None
