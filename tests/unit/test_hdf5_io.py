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
    rgb_ts = np.array([1], dtype=np.uint64)
    depth_ts = np.array([2], dtype=np.uint64)
    recorder.write(
        metadata=metadata,
        rgb_frames=rgb,
        rgb_timestamps=rgb_ts,
        depth_frames=depth,
        depth_timestamps=depth_ts,
        frame_annotations=[{"frame_id": 0}],
        scene_annotations=[{"context": "CORRIDOR"}],
    )

    data = HDF5Reader(path).read()
    assert data["metadata"].session_id == "session-1"
    assert data["metadata"].robot_config == {"platform": "mecanum"}
    np.testing.assert_array_equal(data["rgb_frames"], rgb)
    np.testing.assert_array_equal(data["depth_frames"], depth)
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
    rgb_ts = np.array([100, 200], dtype=np.uint64)
    depth_ts = np.array([110, 210], dtype=np.uint64)
    frame_annotations = [{"frame_id": 0}, {"frame_id": 1}]
    scene_annotations = [{"context": "CORRIDOR"}, {"context": "LOBBY"}]

    recorder.write(
        metadata=metadata,
        rgb_frames=rgb,
        rgb_timestamps=rgb_ts,
        depth_frames=depth,
        depth_timestamps=depth_ts,
        frame_annotations=frame_annotations,
        scene_annotations=scene_annotations,
    )

    batches = list(HDF5Reader(path).iter_batches())

    assert len(batches) == 2
    assert batches[0].rgb_timestamp == 100
    assert batches[0].frame_annotation == {"frame_id": 0}
    assert batches[0].scene_annotation == {"context": "CORRIDOR"}
    assert batches[1].rgb_timestamp == 200
