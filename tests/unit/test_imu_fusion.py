from __future__ import annotations

import numpy as np
import pytest

from src.perception.imu_fusion import IMUFusion


def test_imu_fusion_integrates_velocity_over_time() -> None:
    fusion = IMUFusion()
    accel = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    fusion.update(accel, quat, timestamp_us=1_000_000)
    state = fusion.update(accel, quat, timestamp_us=2_000_000)

    assert state.velocity_xyz_mps[0] == pytest.approx(1.0, abs=0.05)
    assert state.velocity_xyz_mps[1] == pytest.approx(0.0, abs=0.05)
    assert state.heading_rad == pytest.approx(0.0, abs=0.01)


def test_imu_fusion_extracts_heading_from_quaternion() -> None:
    fusion = IMUFusion()
    accel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    yaw_90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)

    state = fusion.update(accel, yaw_90, timestamp_us=1_000_000)

    assert state.heading_rad == pytest.approx(np.pi / 2.0, abs=0.01)


def test_imu_fusion_rejects_wrong_accel_shape() -> None:
    fusion = IMUFusion()
    accel = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="accel_xyz_mps2"):
        fusion.update(accel, quat, timestamp_us=1_000_000)
