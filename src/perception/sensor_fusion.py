from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    ) -> list[FusedEntity]:
        return [
            FusedEntity(
                track_id=track.track_id,
                bbox_xywh=track.bbox_xywh.copy(),
                position_3d=track.position_3d.copy(),
                velocity_3d=track.velocity_3d.copy(),
                heading_rad=0.0,
                confidence=track.confidence,
                nearest_obstacle_distance_m=None,
                nearest_obstacle_centroid_xy=None,
                ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
            )
            for track in tracks
        ]
