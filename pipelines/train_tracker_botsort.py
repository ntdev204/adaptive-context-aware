from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.depth_proc import DepthBoundingBox3D
from src.perception.tracker import MultiObjectTracker, TrackState


@dataclass(frozen=True, slots=True)
class BoTSORTTrainingConfig:
    mot20_dir: Path = Path("data/fine_tuning/mot20")
    hallway_rgbdt_dir: Path = Path("data/fine_tuning/hallway_rgbdt")
    output_path: Path = Path("artifacts/tracker/botsort_tuned.json")
    max_mot20_sequences: int = 2
    max_hallway_sequences: int = 2
    max_frames_per_sequence: int = 200


@dataclass(frozen=True, slots=True)
class BoTSORTTrainingResult:
    best_config_path: Path
    best_score: float
    evaluated_sequences: int


def tune_botsort(config: BoTSORTTrainingConfig = BoTSORTTrainingConfig()) -> BoTSORTTrainingResult:
    mot20_sequences = list(iter_mot20_sequences(config.mot20_dir))[: config.max_mot20_sequences]
    hallway_sequences = list(iter_hallway_sequences(config.hallway_rgbdt_dir))[: config.max_hallway_sequences]
    if not mot20_sequences:
        raise ValueError(f"no MOT20 sequences found under {config.mot20_dir}")
    if not hallway_sequences:
        raise ValueError(f"no hallway_rgbdt sequences found under {config.hallway_rgbdt_dir}")

    candidates = [
        {"iou_threshold": 0.2, "depth_gate_m": 0.8, "max_missed_frames": 2},
        {"iou_threshold": 0.3, "depth_gate_m": 1.0, "max_missed_frames": 3},
        {"iou_threshold": 0.4, "depth_gate_m": 1.2, "max_missed_frames": 4},
    ]

    scored: list[tuple[dict[str, float | int], float]] = []
    for candidate in candidates:
        mot20_score = np.mean(
            [
                evaluate_tracker_on_mot20_sequence(
                    sequence,
                    iou_threshold=float(candidate["iou_threshold"]),
                    depth_gate_m=float(candidate["depth_gate_m"]),
                    max_missed_frames=int(candidate["max_missed_frames"]),
                    max_frames=config.max_frames_per_sequence,
                )
                for sequence in mot20_sequences
            ]
        )
        hallway_score = np.mean(
            [
                evaluate_tracker_on_hallway_sequence(
                    sequence,
                    iou_threshold=float(candidate["iou_threshold"]),
                    depth_gate_m=float(candidate["depth_gate_m"]),
                    max_missed_frames=int(candidate["max_missed_frames"]),
                    max_frames=config.max_frames_per_sequence,
                )
                for sequence in hallway_sequences
            ]
        )
        scored.append((candidate, float(0.7 * mot20_score + 0.3 * hallway_score)))

    best_candidate, best_score = max(scored, key=lambda item: item[1])
    payload = {
        "tracker_type": "bot-sort-compatible-heuristic",
        "mot20_dir": str(config.mot20_dir),
        "hallway_rgbdt_dir": str(config.hallway_rgbdt_dir),
        "evaluated_sequences": {
            "mot20": [sequence["name"] for sequence in mot20_sequences],
            "hallway_rgbdt": [sequence["name"] for sequence in hallway_sequences],
        },
        "best_score": best_score,
        "config": best_candidate,
        "candidates": [{"config": candidate, "score": score} for candidate, score in scored],
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return BoTSORTTrainingResult(
        best_config_path=config.output_path,
        best_score=best_score,
        evaluated_sequences=len(mot20_sequences) + len(hallway_sequences),
    )


def build_tracker(config_path: str | Path) -> MultiObjectTracker:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tracker_config = payload["config"]
    return MultiObjectTracker(
        iou_threshold=float(tracker_config["iou_threshold"]),
        depth_gate_m=float(tracker_config["depth_gate_m"]),
        max_missed_frames=int(tracker_config["max_missed_frames"]),
    )


def iter_mot20_sequences(dataset_dir: Path):
    for split in ("train", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            det_path = sequence_dir / "det" / "det.txt"
            gt_path = sequence_dir / "gt" / "gt.txt"
            seqinfo_path = sequence_dir / "seqinfo.ini"
            if det_path.exists() and gt_path.exists() and seqinfo_path.exists():
                yield {
                    "source": "mot20",
                    "name": sequence_dir.name,
                    "det_path": det_path,
                    "gt_path": gt_path,
                    "seqinfo_path": seqinfo_path,
                }


def iter_hallway_sequences(dataset_dir: Path):
    for scene_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        gt_yaml = scene_dir / "ground_truth_image_plane.yaml"
        if not gt_yaml.exists():
            continue
        ground_truth = json.loads(json.dumps(__import__("yaml").safe_load(gt_yaml.read_text(encoding="utf-8"))))
        for sequence_name, frames in ground_truth.items():
            sequence_dir = scene_dir / sequence_name
            if sequence_dir.exists():
                yield {
                    "source": "hallway_rgbdt",
                    "scene_name": scene_dir.name,
                    "name": sequence_name,
                    "frames": frames,
                    "sequence_dir": sequence_dir,
                }


def evaluate_tracker_on_mot20_sequence(
    sequence: dict[str, Path | str],
    *,
    iou_threshold: float,
    depth_gate_m: float,
    max_missed_frames: int,
    max_frames: int,
) -> float:
    detections_by_frame = _read_mot_rows(Path(sequence["det_path"]))
    gt_by_frame = _read_mot_rows(Path(sequence["gt_path"]), include_track_id=True)
    tracker = MultiObjectTracker(
        iou_threshold=iou_threshold,
        depth_gate_m=depth_gate_m,
        max_missed_frames=max_missed_frames,
    )

    match_hits = 0
    total_gt = 0
    stable_tracks = 0
    seen_track_assignments: dict[int, int] = {}

    for frame_id in sorted(gt_by_frame)[:max_frames]:
        detections = detections_by_frame.get(frame_id, np.zeros((0, 6), dtype=np.float32))
        depth_boxes = _fake_depth_boxes(detections)
        tracks = tracker.update(detections, depth_boxes, delta_time_s=1.0 / 25.0)
        gt_entries = gt_by_frame.get(frame_id, [])
        total_gt += len(gt_entries)
        frame_matches, frame_stable = _score_frame(tracks, gt_entries, seen_track_assignments)
        match_hits += frame_matches
        stable_tracks += frame_stable

    if total_gt == 0:
        return 0.0
    detection_ratio = match_hits / total_gt
    stability_ratio = stable_tracks / total_gt
    return 0.7 * detection_ratio + 0.3 * stability_ratio


def evaluate_tracker_on_hallway_sequence(
    sequence: dict[str, object],
    *,
    iou_threshold: float,
    depth_gate_m: float,
    max_missed_frames: int,
    max_frames: int,
) -> float:
    tracker = MultiObjectTracker(
        iou_threshold=iou_threshold,
        depth_gate_m=depth_gate_m,
        max_missed_frames=max_missed_frames,
    )
    frames = sequence["frames"]
    match_hits = 0
    total_gt = 0
    stable_tracks = 0
    seen_track_assignments: dict[int, int] = {}

    for frame_key in sorted(frames, key=lambda value: int(value))[:max_frames]:
        persons = frames[frame_key]
        detections = []
        gt_entries = []
        for track_id, bbox in persons.items():
            x, y, w, h = [float(value) for value in bbox]
            det = np.array([x, y, w, h, 1.0, 0.0], dtype=np.float32)
            detections.append(det)
            gt_entries.append((int(track_id), det[:4].copy()))
        detections_array = np.vstack(detections) if detections else np.zeros((0, 6), dtype=np.float32)
        depth_boxes = _fake_depth_boxes(detections_array)
        tracks = tracker.update(detections_array, depth_boxes, delta_time_s=1.0 / 30.0)
        total_gt += len(gt_entries)
        frame_matches, frame_stable = _score_frame(tracks, gt_entries, seen_track_assignments)
        match_hits += frame_matches
        stable_tracks += frame_stable

    if total_gt == 0:
        return 0.0
    detection_ratio = match_hits / total_gt
    stability_ratio = stable_tracks / total_gt
    return 0.7 * detection_ratio + 0.3 * stability_ratio


def _score_frame(
    tracks: list[TrackState],
    gt_entries: list[tuple[int, np.ndarray]],
    seen_track_assignments: dict[int, int],
) -> tuple[int, int]:
    matches = 0
    stable = 0
    available_tracks = tracks.copy()
    for gt_track_id, gt_bbox in gt_entries:
        best_track = None
        best_iou = 0.0
        for track in available_tracks:
            iou = _iou(track.bbox_xywh, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None or best_iou < 0.3:
            continue
        matches += 1
        if gt_track_id in seen_track_assignments and seen_track_assignments[gt_track_id] == best_track.track_id:
            stable += 1
        seen_track_assignments[gt_track_id] = best_track.track_id
        available_tracks.remove(best_track)
    return matches, stable


def _read_mot_rows(path: Path, *, include_track_id: bool = False):
    rows: dict[int, list[tuple[int, np.ndarray]] | np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(part) for part in line.split(",")]
        frame_id = int(values[0])
        track_id = int(values[1])
        bbox = np.array([values[2], values[3], values[4], values[5]], dtype=np.float32)
        confidence = float(values[6]) if len(values) > 6 else 1.0
        if include_track_id:
            rows.setdefault(frame_id, []).append((track_id, bbox))
        else:
            det = np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence, 0.0], dtype=np.float32)
            if frame_id not in rows:
                rows[frame_id] = det.reshape(1, 6)
            else:
                rows[frame_id] = np.vstack([rows[frame_id], det])  # type: ignore[index]
    return rows


def _fake_depth_boxes(detections: np.ndarray) -> list[DepthBoundingBox3D]:
    boxes: list[DepthBoundingBox3D] = []
    for detection in detections:
        x, y, w, h, conf, cls = detection
        boxes.append(
            DepthBoundingBox3D(
                x_m=float(x) / 100.0,
                y_m=float(y) / 100.0,
                z_m=3.0,
                width_m=float(w) / 100.0,
                height_m=float(h) / 100.0,
                confidence=float(conf),
                class_id=float(cls),
            )
        )
    return boxes


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    union = aw * ah + bw * bh - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the BoT-SORT-style tracker against MOT20, then hallway_rgbdt.")
    parser.add_argument("--mot20-dir", type=Path, default=BoTSORTTrainingConfig.mot20_dir)
    parser.add_argument("--hallway-rgbdt-dir", type=Path, default=BoTSORTTrainingConfig.hallway_rgbdt_dir)
    parser.add_argument("--output-path", type=Path, default=BoTSORTTrainingConfig.output_path)
    parser.add_argument("--max-mot20-sequences", type=int, default=BoTSORTTrainingConfig.max_mot20_sequences)
    parser.add_argument("--max-hallway-sequences", type=int, default=BoTSORTTrainingConfig.max_hallway_sequences)
    parser.add_argument("--max-frames-per-sequence", type=int, default=BoTSORTTrainingConfig.max_frames_per_sequence)
    args = parser.parse_args()

    result = tune_botsort(
        BoTSORTTrainingConfig(
            mot20_dir=args.mot20_dir,
            hallway_rgbdt_dir=args.hallway_rgbdt_dir,
            output_path=args.output_path,
            max_mot20_sequences=args.max_mot20_sequences,
            max_hallway_sequences=args.max_hallway_sequences,
            max_frames_per_sequence=args.max_frames_per_sequence,
        )
    )
    print(f"best_config={result.best_config_path}")
    print(f"best_score={result.best_score:.4f}")
    print(f"evaluated_sequences={result.evaluated_sequences}")


if __name__ == "__main__":
    main()
