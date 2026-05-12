from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(slots=True)
class SessionMetadata:
    session_id: str
    start_time: int
    duration_s: float
    robot_config: dict[str, Any]
    environment: str


class HDF5Recorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        metadata: SessionMetadata,
        rgb_frames: np.ndarray,
        rgb_timestamps: np.ndarray,
        depth_frames: np.ndarray,
        depth_timestamps: np.ndarray,
        frame_annotations: list[dict[str, Any]] | None = None,
        scene_annotations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as handle:
            meta = handle.create_group("metadata")
            meta.attrs["session_id"] = metadata.session_id
            meta.attrs["start_time"] = metadata.start_time
            meta.attrs["duration_s"] = metadata.duration_s
            meta.attrs["robot_config"] = json.dumps(metadata.robot_config)
            meta.attrs["environment"] = metadata.environment

            rgb = handle.create_group("rgb_frames")
            rgb.create_dataset(
                "data",
                data=rgb_frames,
                chunks=(min(10, len(rgb_frames)), 480, 640, 3),
                compression="gzip",
                compression_opts=4,
            )
            rgb.create_dataset("timestamps", data=rgb_timestamps)

            depth = handle.create_group("depth_frames")
            depth.create_dataset(
                "data",
                data=depth_frames,
                chunks=(min(10, len(depth_frames)), 480, 640),
                compression="gzip",
                compression_opts=4,
            )
            depth.create_dataset("timestamps", data=depth_timestamps)

            if frame_annotations is not None or scene_annotations is not None:
                annotations = handle.create_group("annotations")
                string_dtype = h5py.string_dtype(encoding="utf-8")
                annotations.create_dataset(
                    "frame_annotations",
                    data=np.array([json.dumps(item) for item in frame_annotations or []], dtype=string_dtype),
                )
                annotations.create_dataset(
                    "scene_annotations",
                    data=np.array([json.dumps(item) for item in scene_annotations or []], dtype=string_dtype),
                )
