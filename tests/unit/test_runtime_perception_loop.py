from __future__ import annotations

import numpy as np

from src.perception.sensor_fusion import FusedEntity
from src.runtime.frame_source import _looks_like_jpeg
from src.runtime.perception_loop import build_result_message


def test_build_result_message_maps_fused_entities_to_result_plane() -> None:
    entity = FusedEntity(
        track_id=7,
        bbox_xywh=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        position_3d=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        velocity_3d=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        heading_rad=0.25,
        confidence=0.91,
        nearest_obstacle_distance_m=1.4,
        nearest_obstacle_centroid_xy=np.array([0.0, 0.0], dtype=np.float32),
        ego_velocity_xyz_mps=np.zeros(3, dtype=np.float32),
    )

    message = build_result_message(
        source_id="adaptive-runtime",
        sequence=11,
        timestamp_us=123456,
        entities=[entity],
        timings={"camera_ms": 2.0, "detector_ms": 5.0, "fusion_ms": 1.0, "total_ms": 10.0},
    )

    assert message.source_id == "adaptive-runtime"
    assert message.sequence == 11
    assert message.metrics.fps == 100.0
    assert message.entities[0].track_id == 7
    np.testing.assert_allclose(message.entities[0].bbox_xywh, entity.bbox_xywh)
    np.testing.assert_allclose(message.entities[0].position_xyz_m, entity.position_3d)


def test_scada_frame_filter_accepts_only_jpeg_payloads() -> None:
    assert _looks_like_jpeg(b"\xff\xd8payload\xff\xd9")
    assert not _looks_like_jpeg(b"MAP:\x89PNG")
    assert not _looks_like_jpeg(b"not-a-frame")
