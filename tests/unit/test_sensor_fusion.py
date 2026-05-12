from __future__ import annotations

from src.perception.sensor_fusion import FusedEntity, SensorFusion
from src.perception.tracker import TrackState
import numpy as np


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


def test_sensor_fusion_outputs_complete_entity_fields() -> None:
    fusion = SensorFusion()
    tracks = [_track(track_id=1, x=1.0, y=0.5, z=2.0)]

    fused = fusion.fuse(tracks)

    assert len(fused) == 1
    entity = fused[0]
    assert entity.track_id == 1
    assert entity.position_3d.shape == (3,)
    assert entity.velocity_3d.shape == (3,)
    assert entity.confidence == 0.9


def test_sensor_fusion_empty_tracks() -> None:
    fusion = SensorFusion()
    fused = fusion.fuse([])
    assert fused == []
