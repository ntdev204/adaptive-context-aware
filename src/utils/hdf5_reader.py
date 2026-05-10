from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from .hdf5_recorder import SessionMetadata


class HDF5Reader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        with h5py.File(self.path, "r") as handle:
            metadata = SessionMetadata(
                session_id=handle["metadata"].attrs["session_id"],
                start_time=int(handle["metadata"].attrs["start_time"]),
                duration_s=float(handle["metadata"].attrs["duration_s"]),
                robot_config=json.loads(handle["metadata"].attrs["robot_config"]),
                environment=handle["metadata"].attrs["environment"],
            )
            result: dict[str, Any] = {
                "metadata": metadata,
                "rgb_frames": handle["rgb_frames"]["data"][...],
                "rgb_timestamps": handle["rgb_frames"]["timestamps"][...],
                "depth_frames": handle["depth_frames"]["data"][...],
                "depth_timestamps": handle["depth_frames"]["timestamps"][...],
                "lidar_scans": handle["lidar_scans"]["data"][...],
                "lidar_num_points": handle["lidar_scans"]["num_points"][...],
                "lidar_timestamps": handle["lidar_scans"]["timestamps"][...],
                "imu_accel": handle["imu"]["accel"][...],
                "imu_gyro": handle["imu"]["gyro"][...],
                "imu_quat": handle["imu"]["quat"][...],
                "imu_timestamps": handle["imu"]["timestamps"][...],
            }
            if "annotations" in handle:
                frame_values = handle["annotations"]["frame_annotations"].asstr()[...]
                scene_values = handle["annotations"]["scene_annotations"].asstr()[...]
                result["frame_annotations"] = [json.loads(item) for item in frame_values]
                result["scene_annotations"] = [json.loads(item) for item in scene_values]
            return result
