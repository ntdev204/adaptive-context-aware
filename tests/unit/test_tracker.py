from __future__ import annotations

import numpy as np

from src.perception.depth_proc import DepthBoundingBox3D
from src.perception.tracker import MultiObjectTracker


def _depth_box(x_m: float, y_m: float, z_m: float, confidence: float = 0.9) -> DepthBoundingBox3D:
    return DepthBoundingBox3D(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        width_m=0.4,
        height_m=1.7,
        confidence=confidence,
        class_id=0.0,
    )


def test_tracker_keeps_stable_id_across_frames() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.2, depth_gate_m=0.5)
    detections_frame1 = np.array([[100.0, 80.0, 40.0, 120.0, 0.9, 0.0]], dtype=np.float32)
    detections_frame2 = np.array([[102.0, 82.0, 40.0, 120.0, 0.91, 0.0]], dtype=np.float32)

    tracks_1 = tracker.update(detections_frame1, [_depth_box(0.0, 0.0, 2.0)])
    tracks_2 = tracker.update(detections_frame2, [_depth_box(0.02, 0.01, 2.02)], delta_time_s=0.1)

    assert len(tracks_1) == 1
    assert len(tracks_2) == 1
    assert tracks_1[0].track_id == tracks_2[0].track_id
    assert tracks_2[0].age == 2


def test_tracker_uses_depth_gate_to_split_overlapping_boxes() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.2, depth_gate_m=0.3)
    detections = np.array([[100.0, 80.0, 40.0, 120.0, 0.9, 0.0]], dtype=np.float32)

    tracks_1 = tracker.update(detections, [_depth_box(0.0, 0.0, 2.0)])
    tracks_2 = tracker.update(detections, [_depth_box(0.0, 0.0, 3.0)], delta_time_s=0.1)

    assert len(tracks_1) == 1
    assert len(tracks_2) == 2
    assert tracks_1[0].track_id != tracks_2[-1].track_id


def test_tracker_removes_stale_tracks_after_misses() -> None:
    tracker = MultiObjectTracker(max_missed_frames=1)
    detections = np.array([[100.0, 80.0, 40.0, 120.0, 0.9, 0.0]], dtype=np.float32)

    tracker.update(detections, [_depth_box(0.0, 0.0, 2.0)])
    tracker.update(np.zeros((0, 6), dtype=np.float32), [], delta_time_s=0.1)
    tracks = tracker.update(np.zeros((0, 6), dtype=np.float32), [], delta_time_s=0.1)

    assert tracks == []
