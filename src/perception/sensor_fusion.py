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
    confidence: float


class SensorFusion:
    """Fuse depth-enhanced tracks from RGB-D camera into FusedEntity objects.

    LiDAR and IMU have been removed; obstacle proximity and ego-motion
    estimation are no longer available at this layer.
    """

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
                confidence=track.confidence,
            )
            for track in tracks
        ]
