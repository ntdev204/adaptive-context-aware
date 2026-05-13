from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.depth_proc import CameraIntrinsics, DepthProcessor  # noqa: E402
from src.perception.tracker import MultiObjectTracker, TrackState  # noqa: E402
from src.utils.hdf5_recorder import SessionMetadata  # noqa: E402

FRAME_HEIGHT = 480
FRAME_WIDTH = 640
DEFAULT_FPS = 30.0
DEFAULT_BASE_TIMESTAMP_US = 1_715_000_000_000_000
PERSON_CLASS_ID = 0.0

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
_ROBOFLOW_FRAME_RE = re.compile(r"(?P<clip>.+?_mp4)-(?P<frame>\d+)_jpg", re.IGNORECASE)


class DetectionProvider(Protocol):
    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        """Return detections as float32 [N, 6] in xywh + confidence + class format."""


@dataclass(frozen=True, slots=True)
class SyntheticH5Config:
    raw_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/synthetic_h5")
    model_path: Path = Path("models/fine_tuning/best.pt")
    tracker_config_path: Path = Path("models/fine_tuning/botsort_tuned.json")
    source_kind: str = "all"
    image_detection_source: str = "hybrid"
    confidence_threshold: float = 0.25
    person_class_names: tuple[str, ...] = ("person", "nguoi")
    scene_context: str = "UNKNOWN"
    fps: float = DEFAULT_FPS
    video_stride: int = 1
    max_sequences: int | None = None
    max_frames_per_sequence: int | None = None
    base_timestamp_us: int = DEFAULT_BASE_TIMESTAMP_US
    overwrite: bool = False


DEFAULT_SYNTHETIC_H5_CONFIG = SyntheticH5Config()


@dataclass(frozen=True, slots=True)
class FrameRef:
    path: Path
    source_frame_id: int | None


@dataclass(frozen=True, slots=True)
class ImageSequence:
    name: str
    source_path: Path
    frames: tuple[FrameRef, ...]
    fps: float


@dataclass(frozen=True, slots=True)
class VideoSequence:
    name: str
    source_path: Path
    fps: float
    stride: int


@dataclass(frozen=True, slots=True)
class GeneratedH5Summary:
    path: Path
    source_name: str
    frame_count: int
    annotation_count: int


class YoloDetectionProvider:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        person_class_names: tuple[str, ...],
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.person_class_names = person_class_names
        self._model: object | None = None
        self._class_id: int | None = None

    def detect(self, frame_bgr: np.ndarray, frame_id: int | None = None) -> np.ndarray:
        model = self._get_model()
        class_id = self._get_person_class_id(model)
        results = model.predict(  # type: ignore[union-attr]
            frame_bgr,
            conf=self.confidence_threshold,
            classes=[class_id],
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes.conf.cpu().numpy().astype(np.float32)
        xywh = xyxy.copy()
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        detections = np.column_stack(
            [
                xywh,
                confidences,
                np.full(len(xywh), PERSON_CLASS_ID, dtype=np.float32),
            ]
        )
        return _clip_detections(detections.astype(np.float32), frame_shape=frame_bgr.shape)

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("ultralytics is required to run YOLO inference: pip install -e .[engine]") from exc
            if not self.model_path.exists():
                raise FileNotFoundError(f"missing YOLO model: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def _get_person_class_id(self, model: object) -> int:
        if self._class_id is None:
            names = getattr(model, "names", {})
            self._class_id = resolve_person_class_id(names, self.person_class_names)
        return self._class_id


class StreamingHDF5Writer:
    def __init__(self, path: Path, metadata: SessionMetadata) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(path, "w")
        self.frame_annotations: list[dict[str, object]] = []
        self.scene_annotations: list[dict[str, object]] = []
        self._write_metadata(metadata)
        self._create_datasets()

    def __enter__(self) -> StreamingHDF5Writer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.finalize()
        self.handle.close()

    def append(
        self,
        *,
        rgb_frame: np.ndarray,
        timestamp_us: int,
        depth_frame: np.ndarray,
        lidar_scan: np.ndarray,
        imu_accel: np.ndarray,
        imu_gyro: np.ndarray,
        imu_quat: np.ndarray,
        frame_annotation: dict[str, object],
        scene_annotation: dict[str, object],
    ) -> None:
        index = self.handle["rgb_frames"]["data"].shape[0]
        self._append_dataset("rgb_frames/data", index, rgb_frame)
        self._append_dataset("rgb_frames/timestamps", index, np.uint64(timestamp_us))
        self._append_dataset("depth_frames/data", index, depth_frame)
        self._append_dataset("depth_frames/timestamps", index, np.uint64(timestamp_us))
        self._append_dataset("lidar_scans/data", index, lidar_scan)
        self._append_dataset("lidar_scans/num_points", index, np.uint32(len(lidar_scan)))
        self._append_dataset("lidar_scans/timestamps", index, np.uint64(timestamp_us))
        self._append_dataset("imu/accel", index, imu_accel.astype(np.float32))
        self._append_dataset("imu/gyro", index, imu_gyro.astype(np.float32))
        self._append_dataset("imu/quat", index, imu_quat.astype(np.float32))
        self._append_dataset("imu/timestamps", index, np.uint64(timestamp_us))
        self.frame_annotations.append(frame_annotation)
        self.scene_annotations.append(scene_annotation)

    def finalize(self) -> None:
        if "annotations" in self.handle:
            del self.handle["annotations"]
        annotations = self.handle.create_group("annotations")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        annotations.create_dataset(
            "frame_annotations",
            data=np.array([json.dumps(item) for item in self.frame_annotations], dtype=string_dtype),
        )
        annotations.create_dataset(
            "scene_annotations",
            data=np.array([json.dumps(item) for item in self.scene_annotations], dtype=string_dtype),
        )

    def _write_metadata(self, metadata: SessionMetadata) -> None:
        meta = self.handle.create_group("metadata")
        meta.attrs["session_id"] = metadata.session_id
        meta.attrs["start_time"] = metadata.start_time
        meta.attrs["duration_s"] = metadata.duration_s
        meta.attrs["robot_config"] = json.dumps(metadata.robot_config)
        meta.attrs["environment"] = metadata.environment

    def _create_datasets(self) -> None:
        rgb = self.handle.create_group("rgb_frames")
        rgb.create_dataset(
            "data",
            shape=(0, FRAME_HEIGHT, FRAME_WIDTH, 3),
            maxshape=(None, FRAME_HEIGHT, FRAME_WIDTH, 3),
            chunks=(1, FRAME_HEIGHT, FRAME_WIDTH, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
        rgb.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)

        depth = self.handle.create_group("depth_frames")
        depth.create_dataset(
            "data",
            shape=(0, FRAME_HEIGHT, FRAME_WIDTH),
            maxshape=(None, FRAME_HEIGHT, FRAME_WIDTH),
            chunks=(1, FRAME_HEIGHT, FRAME_WIDTH),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        depth.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)

        lidar = self.handle.create_group("lidar_scans")
        lidar.create_dataset(
            "data",
            shape=(0, 360, 2),
            maxshape=(None, 360, 2),
            chunks=(1, 360, 2),
            dtype=np.float32,
        )
        lidar.create_dataset("num_points", shape=(0,), maxshape=(None,), dtype=np.uint32)
        lidar.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)

        imu = self.handle.create_group("imu")
        imu.create_dataset("accel", shape=(0, 3), maxshape=(None, 3), chunks=(1, 3), dtype=np.float32)
        imu.create_dataset("gyro", shape=(0, 3), maxshape=(None, 3), chunks=(1, 3), dtype=np.float32)
        imu.create_dataset("quat", shape=(0, 4), maxshape=(None, 4), chunks=(1, 4), dtype=np.float32)
        imu.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype=np.uint64)

    def _append_dataset(self, dataset_path: str, index: int, value: np.ndarray | np.generic) -> None:
        dataset = self.handle[dataset_path]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = value


def generate_synthetic_h5_dataset(
    config: SyntheticH5Config = SyntheticH5Config(),
    detector: DetectionProvider | None = None,
) -> list[GeneratedH5Summary]:
    if config.source_kind not in {"all", "images", "videos"}:
        raise ValueError("source_kind must be one of: all, images, videos")
    if config.image_detection_source not in {"yolo", "labels", "hybrid"}:
        raise ValueError("image_detection_source must be one of: yolo, labels, hybrid")
    if config.video_stride < 1:
        raise ValueError("video_stride must be >= 1")

    sequences = _discover_sequences(config)
    needs_detector = config.image_detection_source in {"yolo", "hybrid"} or any(
        isinstance(sequence, VideoSequence) for sequence in sequences
    )
    if detector is None and needs_detector:
        detector = YoloDetectionProvider(
            config.model_path,
            confidence_threshold=config.confidence_threshold,
            person_class_names=config.person_class_names,
        )
    summaries: list[GeneratedH5Summary] = []
    for sequence_index, sequence in enumerate(sequences):
        output_path = config.output_dir / f"{_slugify(sequence.name)}.h5"
        if output_path.exists() and not config.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        tracker = _load_tracker(config.tracker_config_path)
        summary = _write_sequence(sequence, sequence_index, output_path, config, detector, tracker)
        summaries.append(summary)
    return summaries


def resolve_person_class_id(names: object, preferred_names: tuple[str, ...] = ("person", "nguoi")) -> int:
    if isinstance(names, dict):
        name_items = [(int(key), str(value)) for key, value in names.items()]
    elif isinstance(names, (list, tuple)):
        name_items = [(index, str(value)) for index, value in enumerate(names)]
    else:
        return 0

    normalized_preferred = {_normalize_class_name(name) for name in preferred_names}
    for class_id, class_name in name_items:
        if _normalize_class_name(class_name) in normalized_preferred:
            return class_id
    return name_items[0][0] if name_items else 0


def discover_image_sequences(raw_dir: Path, *, fps: float = DEFAULT_FPS) -> list[ImageSequence]:
    sequences: list[ImageSequence] = []
    dataset_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir() and (path / "data.yaml").exists())
    for dataset_dir in dataset_dirs:
        for split in ("train", "valid", "test"):
            images_dir = dataset_dir / split / "images"
            if not images_dir.exists():
                continue
            grouped: dict[str, list[FrameRef]] = {}
            for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in _IMAGE_EXTENSIONS):
                clip_name, frame_id = _parse_roboflow_frame_name(image_path)
                grouped.setdefault(clip_name, []).append(FrameRef(path=image_path, source_frame_id=frame_id))
            for clip_name, frames in sorted(grouped.items()):
                frames = sorted(
                    frames,
                    key=lambda item: (item.source_frame_id is None, item.source_frame_id or 0, item.path.name),
                )
                sequences.append(
                    ImageSequence(
                        name=f"{dataset_dir.name}_{split}_{clip_name}",
                        source_path=images_dir,
                        frames=tuple(frames),
                        fps=fps,
                    )
                )
    return sequences


def discover_video_sequences(raw_dir: Path, *, stride: int = 1) -> list[VideoSequence]:
    return [
        VideoSequence(name=video_path.stem, source_path=video_path, fps=_probe_video_fps(video_path), stride=stride)
        for video_path in sorted(path for path in raw_dir.rglob("*") if path.suffix.lower() in _VIDEO_EXTENSIONS)
    ]


def _discover_sequences(config: SyntheticH5Config) -> list[ImageSequence | VideoSequence]:
    sequences: list[ImageSequence | VideoSequence] = []
    if config.source_kind in {"all", "images"}:
        sequences.extend(discover_image_sequences(config.raw_dir, fps=config.fps))
    if config.source_kind in {"all", "videos"}:
        sequences.extend(discover_video_sequences(config.raw_dir, stride=config.video_stride))
    if config.max_sequences is not None:
        sequences = sequences[: config.max_sequences]
    if not sequences:
        raise FileNotFoundError(f"no raw image/video sequences found under {config.raw_dir}")
    return sequences


def _write_sequence(
    sequence: ImageSequence | VideoSequence,
    sequence_index: int,
    output_path: Path,
    config: SyntheticH5Config,
    detector: DetectionProvider | None,
    tracker: MultiObjectTracker,
) -> GeneratedH5Summary:
    frame_iter = _iter_sequence_frames(sequence, max_frames=config.max_frames_per_sequence)
    flat_depth = np.full((FRAME_HEIGHT, FRAME_WIDTH), 5.0, dtype=np.float32)
    lidar_scan = _make_synthetic_lidar_scan()
    imu_accel = np.zeros(3, dtype=np.float32)
    imu_gyro = np.zeros(3, dtype=np.float32)
    imu_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    depth_processor = DepthProcessor(CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0))
    track_memory: dict[int, np.ndarray] = {}
    previous_timestamp_us: int | None = None
    frame_count = 0
    annotation_count = 0
    fps = max(sequence.fps, 1e-6)

    metadata = SessionMetadata(
        session_id=f"synthetic-{_slugify(sequence.name)}",
        start_time=config.base_timestamp_us + sequence_index * 1_000_000_000,
        duration_s=0.0,
        robot_config={
            "synthetic": True,
            "generator": "pipelines.generate_synthetic_h5",
            "source_path": str(sequence.source_path),
            "detector_model": str(config.model_path),
            "tracker_config": str(config.tracker_config_path),
        },
        environment=config.scene_context,
    )

    with StreamingHDF5Writer(output_path, metadata) as writer:
        for frame_offset, source_frame_id, frame_bgr in frame_iter:
            timestamp_us = _frame_timestamp_us(metadata.start_time, source_frame_id or frame_offset, fps)
            delta_time_s = (
                (timestamp_us - previous_timestamp_us) / 1_000_000.0
                if previous_timestamp_us is not None
                else 1.0 / fps
            )
            previous_timestamp_us = timestamp_us
            detections = _detect_for_frame(
                sequence=sequence,
                frame_offset=frame_offset,
                frame_bgr=frame_bgr,
                detector=detector,
                config=config,
            )
            depth_boxes = depth_processor.detections_to_3d(flat_depth, detections)
            tracks = tracker.update(detections[: len(depth_boxes)], depth_boxes, delta_time_s=max(delta_time_s, 1e-6))
            frame_annotation = _build_frame_annotation(
                frame_id=frame_count,
                timestamp_us=timestamp_us,
                tracks=tracks,
                track_memory=track_memory,
                scene_context=config.scene_context,
            )
            scene_annotation = dict(frame_annotation["scene"])  # type: ignore[arg-type]
            writer.append(
                rgb_frame=frame_bgr,
                timestamp_us=timestamp_us,
                depth_frame=flat_depth,
                lidar_scan=lidar_scan,
                imu_accel=imu_accel,
                imu_gyro=imu_gyro,
                imu_quat=imu_quat,
                frame_annotation=frame_annotation,
                scene_annotation=scene_annotation,
            )
            annotation_count += len(frame_annotation["persons"])  # type: ignore[arg-type]
            frame_count += 1

    _update_duration(output_path, duration_s=frame_count / fps if frame_count else 0.0)
    return GeneratedH5Summary(
        path=output_path,
        source_name=sequence.name,
        frame_count=frame_count,
        annotation_count=annotation_count,
    )


def _iter_sequence_frames(
    sequence: ImageSequence | VideoSequence,
    *,
    max_frames: int | None,
) -> Iterator[tuple[int, int | None, np.ndarray]]:
    if isinstance(sequence, ImageSequence):
        for frame_offset, frame_ref in enumerate(sequence.frames):
            if max_frames is not None and frame_offset >= max_frames:
                break
            yield frame_offset, frame_ref.source_frame_id, _read_image_bgr(frame_ref.path)
        return

    yield from _iter_video_frames(sequence, max_frames=max_frames)


def _detect_for_frame(
    *,
    sequence: ImageSequence | VideoSequence,
    frame_offset: int,
    frame_bgr: np.ndarray,
    detector: DetectionProvider | None,
    config: SyntheticH5Config,
) -> np.ndarray:
    if isinstance(sequence, ImageSequence):
        label_detections = np.zeros((0, 6), dtype=np.float32)
        if config.image_detection_source in {"labels", "hybrid"}:
            label_detections = detections_from_yolo_label(
                sequence.frames[frame_offset].path,
                config.person_class_names,
            )
        if config.image_detection_source == "labels" or label_detections.size > 0:
            return _valid_detections(label_detections)

    if detector is None:
        return np.zeros((0, 6), dtype=np.float32)
    return _valid_detections(detector.detect(frame_bgr, frame_id=frame_offset))


def detections_from_yolo_label(
    image_path: Path,
    person_class_names: tuple[str, ...] = ("person", "nguoi"),
    *,
    confidence: float = 1.0,
) -> np.ndarray:
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    dataset_dir = image_path.parent.parent.parent
    if not label_path.exists() or not (dataset_dir / "data.yaml").exists():
        return np.zeros((0, 6), dtype=np.float32)

    class_id = resolve_person_class_id_from_dataset(dataset_dir / "data.yaml", person_class_names)
    detections: list[list[float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if int(float(parts[0])) != class_id:
            continue
        values = [float(value) for value in parts[1:]]
        bbox = _yolo_values_to_xywh(values)
        if bbox is None:
            continue
        x, y, w, h = bbox
        detections.append([x, y, w, h, confidence, PERSON_CLASS_ID])
    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(detections, dtype=np.float32)


def resolve_person_class_id_from_dataset(dataset_yaml: Path, person_class_names: tuple[str, ...]) -> int:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to read YOLO data.yaml files") from exc
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    return resolve_person_class_id(payload.get("names", []), person_class_names)


def _yolo_values_to_xywh(values: list[float]) -> tuple[float, float, float, float] | None:
    if len(values) == 4:
        cx, cy, w, h = values
        x = (cx - w / 2.0) * FRAME_WIDTH
        y = (cy - h / 2.0) * FRAME_HEIGHT
        return x, y, w * FRAME_WIDTH, h * FRAME_HEIGHT
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x0 = min(xs) * FRAME_WIDTH
        y0 = min(ys) * FRAME_HEIGHT
        x1 = max(xs) * FRAME_WIDTH
        y1 = max(ys) * FRAME_HEIGHT
        return x0, y0, x1 - x0, y1 - y0
    return None


def _iter_video_frames(
    sequence: VideoSequence,
    *,
    max_frames: int | None,
) -> Iterator[tuple[int, int | None, np.ndarray]]:
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to read video sources") from exc

    capture = cv2.VideoCapture(str(sequence.source_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {sequence.source_path}")
    emitted = 0
    source_frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame_id % sequence.stride == 0:
                resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                yield emitted, source_frame_id, resized.astype(np.uint8, copy=False)
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            source_frame_id += 1
    finally:
        capture.release()


def _read_image_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def _load_tracker(config_path: Path) -> MultiObjectTracker:
    if not config_path.exists():
        return MultiObjectTracker()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    tracker_config = payload.get("config", {})
    return MultiObjectTracker(
        iou_threshold=float(tracker_config.get("iou_threshold", 0.3)),
        depth_gate_m=float(tracker_config.get("depth_gate_m", 1.0)),
        max_missed_frames=int(tracker_config.get("max_missed_frames", 3)),
    )


def _build_frame_annotation(
    *,
    frame_id: int,
    timestamp_us: int,
    tracks: list[TrackState],
    track_memory: dict[int, np.ndarray],
    scene_context: str,
) -> dict[str, object]:
    persons: list[dict[str, object]] = []
    speeds: list[float] = []
    for track in tracks:
        if track.missed_frames > 0:
            continue
        x, y, w, h = [float(value) for value in track.bbox_xywh]
        center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        previous_center = track_memory.get(track.track_id, center)
        delta = center - previous_center
        track_memory[track.track_id] = center
        speed = float(np.linalg.norm(delta))
        speeds.append(speed)
        persons.append(
            {
                "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                "track_id": int(track.track_id),
                "activity": _activity_from_speed(speed),
                "intent_direction": _direction_from_delta(delta),
                "trajectory_pred": _trajectory_from_delta(center, delta),
                "is_anomaly": False,
                "confidence": round(float(np.clip(track.confidence, 0.0, 1.0)), 4),
            }
        )

    scene = {
        "context": scene_context,
        "crowd_density": round(float(min(len(persons) / 20.0, 1.0)), 4),
        "motion_entropy": round(float(min((np.mean(speeds) if speeds else 0.0) / 30.0, 1.0)), 4),
        "anomaly_flag": False,
    }
    return {
        "frame_id": frame_id,
        "timestamp": int(timestamp_us),
        "persons": persons,
        "scene": scene,
    }


def _activity_from_speed(speed_px_per_frame: float) -> str:
    if speed_px_per_frame < 2.0:
        return "STANDING"
    if speed_px_per_frame < 24.0:
        return "WALKING"
    return "RUNNING"


def _direction_from_delta(delta: np.ndarray) -> str:
    dx = float(delta[0])
    dy = float(delta[1])
    if abs(dx) < 2.0 and abs(dy) < 2.0:
        return "STATIONARY"
    horizontal = "EAST" if dx > 2.0 else "WEST" if dx < -2.0 else ""
    vertical = "SOUTH" if dy > 2.0 else "NORTH" if dy < -2.0 else ""
    if vertical and horizontal:
        return {"NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW"}[vertical + horizontal]
    return vertical or horizontal or "STATIONARY"


def _trajectory_from_delta(center: np.ndarray, delta: np.ndarray) -> list[list[float]]:
    return [
        [round(float(center[0] + delta[0] * step), 3), round(float(center[1] + delta[1] * step), 3)]
        for step in (1, 2)
    ]


def _valid_detections(detections: np.ndarray) -> np.ndarray:
    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape [N, 6]")
    keep = (detections[:, 2] > 1.0) & (detections[:, 3] > 1.0) & (detections[:, 4] > 0.0)
    return _clip_detections(detections[keep].copy(), frame_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3))


def _clip_detections(detections: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    height, width, _ = frame_shape
    if detections.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    detections[:, 0] = np.clip(detections[:, 0], 0.0, float(width))
    detections[:, 1] = np.clip(detections[:, 1], 0.0, float(height))
    detections[:, 2] = np.clip(detections[:, 2], 0.0, float(width) - detections[:, 0])
    detections[:, 3] = np.clip(detections[:, 3], 0.0, float(height) - detections[:, 1])
    return detections.astype(np.float32, copy=False)


def _make_synthetic_lidar_scan() -> np.ndarray:
    scan = np.zeros((360, 2), dtype=np.float32)
    scan[:, 0] = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False, dtype=np.float32)
    scan[:, 1] = 5.0
    return scan


def _parse_roboflow_frame_name(path: Path) -> tuple[str, int | None]:
    match = _ROBOFLOW_FRAME_RE.search(path.name)
    if match is None:
        return path.stem, None
    return match.group("clip"), int(match.group("frame"))


def _frame_timestamp_us(start_time_us: int, frame_index: int, fps: float) -> int:
    return int(start_time_us + round(frame_index * 1_000_000.0 / fps))


def _probe_video_fps(path: Path) -> float:
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return DEFAULT_FPS
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return fps if np.isfinite(fps) and fps > 0 else DEFAULT_FPS
    finally:
        capture.release()


def _update_duration(path: Path, *, duration_s: float) -> None:
    with h5py.File(path, "a") as handle:
        handle["metadata"].attrs["duration_s"] = float(duration_s)


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or f"sequence_{int(time.time())}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HDF5 synthetic/pseudo-labeled datasets from data/raw.")
    defaults = DEFAULT_SYNTHETIC_H5_CONFIG
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--tracker-config", type=Path, default=defaults.tracker_config_path)
    parser.add_argument("--source-kind", choices=["all", "images", "videos"], default=defaults.source_kind)
    parser.add_argument(
        "--image-detection-source",
        choices=["yolo", "labels", "hybrid"],
        default=defaults.image_detection_source,
        help="For image folders, use YOLO predictions, existing YOLO labels, or labels with YOLO fallback.",
    )
    parser.add_argument("--conf", type=float, default=defaults.confidence_threshold)
    parser.add_argument("--person-class-name", action="append", dest="person_class_names")
    parser.add_argument("--scene-context", default=defaults.scene_context)
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument("--video-stride", type=int, default=defaults.video_stride)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames-per-sequence", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    person_class_names = tuple(args.person_class_names or defaults.person_class_names)
    summaries = generate_synthetic_h5_dataset(
        SyntheticH5Config(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            tracker_config_path=args.tracker_config,
            source_kind=args.source_kind,
            image_detection_source=args.image_detection_source,
            confidence_threshold=args.conf,
            person_class_names=person_class_names,
            scene_context=args.scene_context,
            fps=args.fps,
            video_stride=args.video_stride,
            max_sequences=args.max_sequences,
            max_frames_per_sequence=args.max_frames_per_sequence,
            overwrite=args.overwrite,
        )
    )
    for summary in summaries:
        print(
            f"generated={summary.path} source={summary.source_name} "
            f"frames={summary.frame_count} annotations={summary.annotation_count}"
        )


if __name__ == "__main__":
    main()
