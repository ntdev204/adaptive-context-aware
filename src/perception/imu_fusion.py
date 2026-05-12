from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EgoMotionState:
    velocity_xyz_mps: np.ndarray
    heading_rad: float
    timestamp_us: int


class IMUFusion:
    """Phase 1 ego-motion baseline.

    This is a lightweight integrator wrapper that keeps the interface stable for a
    future EKF implementation. It estimates ego velocity from linear acceleration
    and heading from the latest quaternion yaw component.
    """

    def __init__(self) -> None:
        self._state = EgoMotionState(
            velocity_xyz_mps=np.zeros(3, dtype=np.float32),
            heading_rad=0.0,
            timestamp_us=0,
        )

    def update(self, accel_xyz_mps2: np.ndarray, quat_xyzw: np.ndarray, timestamp_us: int) -> EgoMotionState:
        accel = self._validate_vector(accel_xyz_mps2, expected_length=3, name="accel_xyz_mps2")
        quat = self._validate_vector(quat_xyzw, expected_length=4, name="quat_xyzw")
        dt_s = 0.0
        if self._state.timestamp_us > 0:
            dt_s = max((timestamp_us - self._state.timestamp_us) / 1_000_000.0, 0.0)

        velocity = self._state.velocity_xyz_mps + accel * dt_s
        heading = self._yaw_from_quaternion(quat)
        self._state = EgoMotionState(
            velocity_xyz_mps=velocity.astype(np.float32),
            heading_rad=float(heading),
            timestamp_us=timestamp_us,
        )
        return self.state()

    def state(self) -> EgoMotionState:
        return EgoMotionState(
            velocity_xyz_mps=self._state.velocity_xyz_mps.copy(),
            heading_rad=self._state.heading_rad,
            timestamp_us=self._state.timestamp_us,
        )

    @staticmethod
    def _validate_vector(vector: np.ndarray, expected_length: int, name: str) -> np.ndarray:
        if vector.shape != (expected_length,):
            raise ValueError(f"expected {name} shape ({expected_length},)")
        if vector.dtype != np.float32:
            raise ValueError(f"expected {name} dtype float32")
        return vector

    @staticmethod
    def _yaw_from_quaternion(quat_xyzw: np.ndarray) -> float:
        x, y, z, w = quat_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))
