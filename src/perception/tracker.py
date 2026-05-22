from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .depth_proc import DepthBoundingBox3D


def _empty_contour_xy() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _empty_contour_points_xyz_m() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox_xywh: np.ndarray
    position_3d: np.ndarray
    velocity_3d: np.ndarray
    age: int
    missed_frames: int
    confidence: float
    class_id: float = 0.0
    contour_xy: np.ndarray = field(default_factory=_empty_contour_xy)
    contour_points_xyz_m: np.ndarray = field(default_factory=_empty_contour_points_xyz_m)


class MultiObjectTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        depth_gate_m: float = 1.0,
        max_missed_frames: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.depth_gate_m = depth_gate_m
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: list[TrackState] = []

    def update(
        self,
        detections: np.ndarray,
        depth_boxes: list[DepthBoundingBox3D],
        delta_time_s: float = 1.0,
    ) -> list[TrackState]:
        if detections.shape[0] != len(depth_boxes):
            raise ValueError("detections and depth_boxes must have the same number of entries")

        matches, unmatched_tracks, unmatched_detections = self._associate(detections, depth_boxes)

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            depth_box = depth_boxes[detection_index]
            new_position = np.array([depth_box.x_m, depth_box.y_m, depth_box.z_m], dtype=np.float32)
            velocity = (new_position - track.position_3d) / max(delta_time_s, 1e-6)
            track.bbox_xywh = detection[:4].copy()
            track.position_3d = new_position
            track.velocity_3d = velocity.astype(np.float32)
            track.age += 1
            track.missed_frames = 0
            track.confidence = float(detection[4])
            track.class_id = float(depth_box.class_id)
            track.contour_xy = depth_box.contour_xy.copy()
            track.contour_points_xyz_m = depth_box.contour_points_xyz_m.copy()

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed_frames += 1

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            depth_box = depth_boxes[detection_index]
            position = np.array([depth_box.x_m, depth_box.y_m, depth_box.z_m], dtype=np.float32)
            self._tracks.append(
                TrackState(
                    track_id=self._next_track_id,
                    bbox_xywh=detection[:4].copy(),
                    position_3d=position,
                    velocity_3d=np.zeros(3, dtype=np.float32),
                    age=1,
                    missed_frames=0,
                    confidence=float(detection[4]),
                    class_id=float(depth_box.class_id),
                    contour_xy=depth_box.contour_xy.copy(),
                    contour_points_xyz_m=depth_box.contour_points_xyz_m.copy(),
                )
            )
            self._next_track_id += 1

        self._tracks = [track for track in self._tracks if track.missed_frames <= self.max_missed_frames]
        return [self._clone_track(track) for track in self._tracks]

    def _associate(
        self,
        detections: np.ndarray,
        depth_boxes: list[DepthBoundingBox3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks:
            return [], [], list(range(len(depth_boxes)))

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, depth_box in enumerate(depth_boxes):
                depth_distance = abs(track.position_3d[2] - depth_box.z_m)
                if depth_distance > self.depth_gate_m:
                    continue
                iou = self._iou(track.bbox_xywh, detections[detection_index, :4])
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_index, detection_index))

        candidate_pairs.sort(reverse=True, key=lambda item: item[0])
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, track_index, detection_index in candidate_pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            matches.append((track_index, detection_index))

        unmatched_tracks = [index for index in range(len(self._tracks)) if index not in matched_tracks]
        unmatched_detections = [index for index in range(len(depth_boxes)) if index not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    @staticmethod
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
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    @staticmethod
    def _clone_track(track: TrackState) -> TrackState:
        return TrackState(
            track_id=track.track_id,
            bbox_xywh=track.bbox_xywh.copy(),
            position_3d=track.position_3d.copy(),
            velocity_3d=track.velocity_3d.copy(),
            age=track.age,
            missed_frames=track.missed_frames,
            confidence=track.confidence,
            class_id=track.class_id,
            contour_xy=track.contour_xy.copy(),
            contour_points_xyz_m=track.contour_points_xyz_m.copy(),
        )
