from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from .hdf5_recorder import SessionMetadata


@dataclass(frozen=True, slots=True)
class HDF5ReplayBatch:
    rgb_frame: Any
    rgb_timestamp: int
    depth_frame: Any
    depth_timestamp: int
    lidar_scan: Any
    lidar_num_points: int
    lidar_timestamp: int
    imu_sample: dict[str, Any]
    frame_annotation: dict[str, Any] | None
    scene_annotation: dict[str, Any] | None


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

    def iter_batches(self) -> Iterator[HDF5ReplayBatch]:
        with h5py.File(self.path, "r") as handle:
            rgb_frames = handle["rgb_frames"]["data"]
            rgb_timestamps = handle["rgb_frames"]["timestamps"]
            depth_frames = handle["depth_frames"]["data"]
            depth_timestamps = handle["depth_frames"]["timestamps"]
            lidar_scans = handle["lidar_scans"]["data"]
            lidar_num_points = handle["lidar_scans"]["num_points"]
            lidar_timestamps = handle["lidar_scans"]["timestamps"]
            imu_accel = handle["imu"]["accel"]
            imu_gyro = handle["imu"]["gyro"]
            imu_quat = handle["imu"]["quat"]
            imu_timestamps = handle["imu"]["timestamps"]

            frame_annotations = []
            scene_annotations = []
            if "annotations" in handle:
                frame_raw = handle["annotations"]["frame_annotations"].asstr()[...]
                scene_raw = handle["annotations"]["scene_annotations"].asstr()[...]
                frame_annotations = [json.loads(item) for item in frame_raw]
                scene_annotations = [json.loads(item) for item in scene_raw]

            frame_count = len(rgb_frames)
            if not (
                len(rgb_timestamps)
                == len(depth_frames)
                == len(depth_timestamps)
                == len(lidar_scans)
                == len(lidar_num_points)
                == len(lidar_timestamps)
                == frame_count
            ):
                raise ValueError("rgb, depth, and lidar records must share the same frame count")

            imu_index = 0
            for frame_index in range(frame_count):
                while (
                    imu_index + 1 < len(imu_timestamps) and imu_timestamps[imu_index + 1] <= rgb_timestamps[frame_index]
                ):
                    imu_index += 1

                frame_annotation = frame_annotations[frame_index] if frame_index < len(frame_annotations) else None
                scene_annotation = scene_annotations[frame_index] if frame_index < len(scene_annotations) else None

                yield HDF5ReplayBatch(
                    rgb_frame=rgb_frames[frame_index],
                    rgb_timestamp=int(rgb_timestamps[frame_index]),
                    depth_frame=depth_frames[frame_index],
                    depth_timestamp=int(depth_timestamps[frame_index]),
                    lidar_scan=lidar_scans[frame_index][: int(lidar_num_points[frame_index])],
                    lidar_num_points=int(lidar_num_points[frame_index]),
                    lidar_timestamp=int(lidar_timestamps[frame_index]),
                    imu_sample={
                        "accel": imu_accel[imu_index].copy(),
                        "gyro": imu_gyro[imu_index].copy(),
                        "quat": imu_quat[imu_index].copy(),
                        "timestamp": int(imu_timestamps[imu_index]),
                    },
                    frame_annotation=frame_annotation,
                    scene_annotation=scene_annotation,
                )
