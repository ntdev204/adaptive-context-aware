from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from jsonschema import validate
from PIL import Image

from pipelines.generate_synthetic_h5 import (
    SyntheticH5Config,
    detections_from_yolo_label,
    generate_synthetic_h5_dataset,
    resolve_person_class_id,
)
from src.utils.hdf5_reader import HDF5Reader


class FakeDetector:
    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        offset = float(frame_id or 0)
        return np.array([[100.0 + offset, 120.0, 80.0, 160.0, 0.91, 0.0]], dtype=np.float32)


def _write_raw_image_dataset(root: Path) -> Path:
    dataset_dir = root / "custom_1"
    images_dir = dataset_dir / "train" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 2\nnames: ['door', 'nguoi']\n",
        encoding="utf-8",
    )
    for frame_id in (1, 2, 3):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 1] = frame_id * 30
        image_path = images_dir / f"clip_a_mp4-{frame_id:04d}_jpg.rf.testhash.jpg"
        Image.fromarray(image).save(image_path)
        label_path = dataset_dir / "train" / "labels" / f"{image_path.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("1 0.5 0.5 0.25 0.5\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    return dataset_dir


def _write_loose_image_sequence(root: Path) -> Path:
    sequence_dir = root / "phone_clip_a"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in (1, 2):
        image = np.full((48, 64, 3), fill_value=frame_id * 20, dtype=np.uint8)
        Image.fromarray(image).save(sequence_dir / f"frame_{frame_id:04d}.jpg")
    return sequence_dir


def _write_partial_h5(root: Path) -> Path:
    path = root / "partial.h5"
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["session_id"] = "partial-session"
        metadata.attrs["start_time"] = 1715000000000000
        metadata.attrs["duration_s"] = 2.0
        metadata.attrs["robot_config"] = json.dumps({"source": "phone"})
        metadata.attrs["environment"] = "UNKNOWN"

        rgb = handle.create_group("rgb_frames")
        rgb.create_dataset("data", data=np.zeros((2, 480, 640, 3), dtype=np.uint8))
        rgb.create_dataset("timestamps", data=np.array([10, 20], dtype=np.uint64))
    return path


def test_generate_synthetic_h5_from_raw_images_uses_existing_labels(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "synthetic_h5"
    _write_raw_image_dataset(raw_dir)
    tracker_config = tmp_path / "botsort_tuned.json"
    tracker_config.write_text(
        json.dumps({"config": {"iou_threshold": 0.2, "depth_gate_m": 1.2, "max_missed_frames": 2}}),
        encoding="utf-8",
    )

    summaries = generate_synthetic_h5_dataset(
        SyntheticH5Config(
            raw_dir=raw_dir,
            output_dir=output_dir,
            tracker_config_path=tracker_config,
            source_kind="images",
            image_detection_source="labels",
            max_sequences=1,
            max_frames_per_sequence=3,
            overwrite=True,
        ),
    )

    assert len(summaries) == 1
    assert summaries[0].frame_count == 3
    assert summaries[0].annotation_count == 3

    data = HDF5Reader(summaries[0].path).read()
    assert data["rgb_frames"].shape == (3, 480, 640, 3)
    assert data["depth_frames"].shape == (3, 480, 640)
    assert data["lidar_scans"].shape == (3, 360, 2)
    assert len(data["frame_annotations"]) == 3
    assert data["frame_annotations"][0]["persons"][0]["track_id"] == 1
    assert data["frame_annotations"][2]["persons"][0]["track_id"] == 1
    assert data["metadata"].robot_config["derived_by"] == ["yolo", "botsort"]

    schema_path = Path(__file__).resolve().parents[2] / "config" / "schemas" / "annotation_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(instance=data["frame_annotations"][0], schema=schema)


def test_generate_synthetic_h5_from_raw_images_can_use_detector(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _write_raw_image_dataset(raw_dir)

    summaries = generate_synthetic_h5_dataset(
        SyntheticH5Config(
            raw_dir=raw_dir,
            output_dir=tmp_path / "synthetic_detector_h5",
            source_kind="images",
            image_detection_source="yolo",
            max_sequences=1,
            max_frames_per_sequence=2,
            overwrite=True,
        ),
        detector=FakeDetector(),
    )

    data = HDF5Reader(summaries[0].path).read()
    assert len(data["frame_annotations"][0]["persons"]) == 1


def test_generate_synthetic_h5_from_loose_images_uses_nan_missing_modalities(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _write_loose_image_sequence(raw_dir)

    summaries = generate_synthetic_h5_dataset(
        SyntheticH5Config(
            raw_dir=raw_dir,
            output_dir=tmp_path / "synthetic",
            source_kind="images",
            image_detection_source="yolo",
            max_sequences=1,
            max_frames_per_sequence=2,
            overwrite=True,
        ),
        detector=FakeDetector(),
    )

    data = HDF5Reader(summaries[0].path).read()
    assert np.isnan(data["depth_frames"]).all()
    assert np.isnan(data["lidar_scans"]).all()
    assert np.isnan(data["imu_accel"]).all()
    assert data["metadata"].robot_config["missing_modalities"] == ["depth", "lidar", "imu"]
    assert data["metadata"].robot_config["derived_by"] == ["yolo", "botsort"]


def test_generate_synthetic_h5_normalizes_partial_h5_sources(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial_h5 = _write_partial_h5(raw_dir)

    summaries = generate_synthetic_h5_dataset(
        SyntheticH5Config(
            raw_dir=raw_dir,
            output_dir=tmp_path / "synthetic",
            source_kind="h5",
            overwrite=True,
        )
    )

    assert len(summaries) == 1
    data = HDF5Reader(summaries[0].path).read()
    assert data["metadata"].robot_config["source_kind"] == "h5"
    np.testing.assert_array_equal(data["rgb_frames"], np.zeros((2, 480, 640, 3), dtype=np.uint8))
    assert np.isnan(data["depth_frames"]).all()
    assert np.isnan(data["lidar_scans"]).all()
    assert np.isnan(data["imu_accel"]).all()
    assert data["metadata"].robot_config["source_path"] == str(partial_h5)
    assert data["metadata"].robot_config["derived_by"] == ["replay", "metadata_normalization"]


def test_detections_from_yolo_label_supports_bbox_and_segmentation_labels(tmp_path) -> None:
    dataset_dir = _write_raw_image_dataset(tmp_path / "raw")
    image_path = next((dataset_dir / "train" / "images").glob("*.jpg"))
    label_path = dataset_dir / "train" / "labels" / f"{image_path.stem}.txt"
    label_path.write_text("1 0.25 0.25 0.50 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n", encoding="utf-8")

    detections = detections_from_yolo_label(image_path, ("nguoi",))

    assert detections.shape == (1, 6)
    np.testing.assert_allclose(detections[0, :4], [160.0, 120.0, 320.0, 240.0])


def test_resolve_person_class_id_supports_vietnamese_person_name() -> None:
    assert resolve_person_class_id({0: "Cua", 4: "nguoi"}) == 4
    assert resolve_person_class_id(["wall", "person"]) == 1
