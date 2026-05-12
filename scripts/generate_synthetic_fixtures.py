from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.utils.hdf5_recorder import HDF5Recorder, SessionMetadata

    root = ROOT
    output = root / "tests" / "fixtures" / "sample_recording.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    images_dir = root / "tests" / "fixtures" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((2, 480, 640, 3), dtype=np.uint8)
    depth = np.ones((2, 480, 640), dtype=np.float32)
    lidar = np.zeros((2, 360, 2), dtype=np.float32)
    lidar[:, :, 0] = np.linspace(0.0, 6.28, 360, dtype=np.float32)
    lidar[:, :, 1] = 1.0
    imu_accel = np.zeros((4, 3), dtype=np.float32)
    imu_gyro = np.zeros((4, 3), dtype=np.float32)
    imu_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1))

    HDF5Recorder(output).write(
        metadata=SessionMetadata(
            session_id="synthetic-session",
            start_time=1715000000000000,
            duration_s=5.0,
            robot_config={"platform": "mecanum", "synthetic": True},
            environment="test_lab",
        ),
        rgb_frames=rgb,
        rgb_timestamps=np.array([1, 2], dtype=np.uint64),
        depth_frames=depth,
        depth_timestamps=np.array([1, 2], dtype=np.uint64),
        lidar_scans=lidar,
        lidar_num_points=np.array([360, 360], dtype=np.uint32),
        lidar_timestamps=np.array([1, 2], dtype=np.uint64),
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        imu_quat=imu_quat,
        imu_timestamps=np.array([1, 2, 3, 4], dtype=np.uint64),
        frame_annotations=[{"frame_id": 0}, {"frame_id": 1}],
        scene_annotations=[{"context": "UNKNOWN"}, {"context": "UNKNOWN"}],
    )

    for index in range(5):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 1] = 40 * index
        image[40:200, 120:200, 2] = 180
        Image.fromarray(image).save(images_dir / f"person_{index:02d}.png")


if __name__ == "__main__":
    main()
