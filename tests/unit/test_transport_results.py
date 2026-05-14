from __future__ import annotations

import numpy as np

from src.transport.results import (
    PerceptionResultCodec,
    PerceptionResultMessage,
    RuntimeMetricsMessage,
    TrackedEntityMessage,
)


def test_perception_result_roundtrip_uses_protobuf_bytes() -> None:
    message = PerceptionResultMessage(
        source_id="jetson-100",
        sequence=42,
        timestamp_us=123456,
        entities=[
            TrackedEntityMessage(
                track_id=1,
                bbox_xywh=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
                position_xyz_m=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                velocity_xyz_mps=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                heading_rad=0.5,
                confidence=0.9,
                nearest_obstacle_distance_m=1.2,
            )
        ],
        metrics=RuntimeMetricsMessage(
            total_latency_ms=12.0,
            camera_latency_ms=2.0,
            detector_latency_ms=7.0,
            fusion_latency_ms=1.0,
            fps=30.0,
        ),
    )

    raw = PerceptionResultCodec.encode(message)
    decoded = PerceptionResultCodec.decode(raw)

    assert isinstance(raw, bytes)
    assert decoded.source_id == "jetson-100"
    assert decoded.entities[0].track_id == 1
    np.testing.assert_allclose(decoded.entities[0].position_xyz_m, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert decoded.metrics.fps == 30.0
