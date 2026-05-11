from __future__ import annotations

import numpy as np

from src.utils.hdf5_reader import HDF5Reader
from src.utils.hdf5_recorder import HDF5Recorder, SessionMetadata


def test_hdf5_roundtrip(tmp_path) -> None:
    path = tmp_path / "sample.h5"
    recorder = HDF5Recorder(path)
    metadata = SessionMetadata(
        session_id="session-1",
        start_time=1715000000000000,
        duration_s=5.0,
        robot_config={"platform": "mecanum"},
        environment="corridor_floor2",
    )
    rgb = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    depth = np.ones((1, 480, 640), dtype=np.float32)
    lidar = np.zeros((1, 360, 2), dtype=np.float32)
    lidar[0, 0] = [0.0, 1.5]
    imu_accel = np.zeros((2, 3), dtype=np.float32)
    imu_gyro = np.zeros((2, 3), dtype=np.float32)
    imu_quat = np.zeros((2, 4), dtype=np.float32)
    rgb_ts = np.array([1], dtype=np.uint64)
    depth_ts = np.array([2], dtype=np.uint64)
    lidar_num_points = np.array([1], dtype=np.uint32)
    lidar_ts = np.array([3], dtype=np.uint64)
    imu_ts = np.array([4, 5], dtype=np.uint64)
    recorder.write(
        metadata=metadata,
        rgb_frames=rgb,
        rgb_timestamps=rgb_ts,
        depth_frames=depth,
        depth_timestamps=depth_ts,
        lidar_scans=lidar,
        lidar_num_points=lidar_num_points,
        lidar_timestamps=lidar_ts,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        imu_quat=imu_quat,
        imu_timestamps=imu_ts,
        frame_annotations=[{"frame_id": 0}],
        scene_annotations=[{"context": "CORRIDOR"}],
    )

    data = HDF5Reader(path).read()
    assert data["metadata"].session_id == "session-1"
    assert data["metadata"].robot_config == {"platform": "mecanum"}
    np.testing.assert_array_equal(data["rgb_frames"], rgb)
    np.testing.assert_array_equal(data["depth_frames"], depth)
    np.testing.assert_array_equal(data["lidar_num_points"], lidar_num_points)
    assert data["frame_annotations"] == [{"frame_id": 0}]


def test_hdf5_replay_iterator_aligns_modalities(tmp_path) -> None:
    path = tmp_path / "sample.h5"
    recorder = HDF5Recorder(path)
    metadata = SessionMetadata(
        session_id="session-2",
        start_time=1715000000000000,
        duration_s=5.0,
        robot_config={"platform": "mecanum"},
        environment="lobby",
    )
    rgb = np.stack([np.full((480, 640, 3), fill_value=index, dtype=np.uint8) for index in range(2)])
    depth = np.stack([np.full((480, 640), fill_value=float(index), dtype=np.float32) for index in range(2)])
    lidar = np.zeros((2, 360, 2), dtype=np.float32)
    lidar[0, :2] = [[0.0, 1.0], [0.1, 1.1]]
    lidar[1, :1] = [[0.0, 2.0]]
    imu_accel = np.zeros((4, 3), dtype=np.float32)
    imu_gyro = np.zeros((4, 3), dtype=np.float32)
    imu_quat = np.zeros((4, 4), dtype=np.float32)
    rgb_ts = np.array([100, 200], dtype=np.uint64)
    depth_ts = np.array([110, 210], dtype=np.uint64)
    lidar_num_points = np.array([2, 1], dtype=np.uint32)
    lidar_ts = np.array([120, 220], dtype=np.uint64)
    imu_ts = np.array([90, 115, 205, 225], dtype=np.uint64)
    frame_annotations = [{"frame_id": 0}, {"frame_id": 1}]
    scene_annotations = [{"context": "CORRIDOR"}, {"context": "LOBBY"}]

    recorder.write(
        metadata=metadata,
        rgb_frames=rgb,
        rgb_timestamps=rgb_ts,
        depth_frames=depth,
        depth_timestamps=depth_ts,
        lidar_scans=lidar,
        lidar_num_points=lidar_num_points,
        lidar_timestamps=lidar_ts,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        imu_quat=imu_quat,
        imu_timestamps=imu_ts,
        frame_annotations=frame_annotations,
        scene_annotations=scene_annotations,
    )

    batches = list(HDF5Reader(path).iter_batches())

    assert len(batches) == 2
    assert batches[0].rgb_timestamp == 100
    assert batches[0].lidar_num_points == 2
    assert batches[0].frame_annotation == {"frame_id": 0}
    assert batches[0].scene_annotation == {"context": "CORRIDOR"}
    assert batches[1].rgb_timestamp == 200
    assert batches[1].lidar_num_points == 1
    assert batches[1].imu_sample["timestamp"] == 115
