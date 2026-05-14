from __future__ import annotations

import numpy as np

from src.transport.messages import ImuSampleMessage, LidarScanMessage, SensorMessageCodec


def test_lidar_message_roundtrip_uses_protobuf_bytes() -> None:
    message = LidarScanMessage(
        source_id="pi-101",
        sequence=7,
        timestamp_us=123456,
        scan_points=np.array([[0.0, 1.0], [0.1, 1.1]], dtype=np.float32),
    )

    raw = SensorMessageCodec.encode(message)
    decoded = SensorMessageCodec.decode(raw)

    assert isinstance(raw, bytes)
    assert isinstance(decoded, LidarScanMessage)
    assert decoded.source_id == "pi-101"
    np.testing.assert_allclose(decoded.scan_points, message.scan_points)


def test_imu_message_roundtrip() -> None:
    message = ImuSampleMessage(
        source_id="pi-101",
        sequence=8,
        timestamp_us=123999,
        accel_xyz_mps2=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    decoded = SensorMessageCodec.decode(SensorMessageCodec.encode(message))

    assert isinstance(decoded, ImuSampleMessage)
    np.testing.assert_allclose(decoded.accel_xyz_mps2, message.accel_xyz_mps2)
    np.testing.assert_allclose(decoded.quat_xyzw, message.quat_xyzw)
