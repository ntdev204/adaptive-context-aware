from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.constants import DEPTH_SHAPE_HW
from src.utils.validation import validate_ndarray


def _empty_contour_xy() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _empty_contour_points_xyz_m() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class DepthBoundingBox3D:
    x_m: float
    y_m: float
    z_m: float
    width_m: float
    height_m: float
    confidence: float
    class_id: float = 0.0
    contour_xy: np.ndarray = field(default_factory=_empty_contour_xy)
    contour_points_xyz_m: np.ndarray = field(default_factory=_empty_contour_points_xyz_m)


class DepthProcessor:
    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    def detections_to_3d(
        self,
        depth_map_m: np.ndarray,
        detections: np.ndarray,
    ) -> list[DepthBoundingBox3D]:
        self._validate_depth_map(depth_map_m)
        self._validate_detections(detections)

        if detections.shape[0] == 0:
            return []

        results: list[DepthBoundingBox3D] = []
        fx, fy, cx, cy = self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy
        for x, y, w, h, conf, cls in detections:
            depth_value = self._roi_depth(depth_map_m, x, y, w, h)
            if depth_value is None:
                continue
            center_x = x + w * 0.5
            center_y = y + h * 0.5
            contour_xy = self._roi_contour(depth_map_m, x, y, w, h)
            contour_points_xyz_m = self._contour_points_to_3d(contour_xy, depth_map_m)
            results.append(
                DepthBoundingBox3D(
                    x_m=(center_x - cx) * depth_value / fx,
                    y_m=(center_y - cy) * depth_value / fy,
                    z_m=depth_value,
                    width_m=w * depth_value / fx,
                    height_m=h * depth_value / fy,
                    confidence=float(conf),
                    class_id=float(cls),
                    contour_xy=contour_xy,
                    contour_points_xyz_m=contour_points_xyz_m,
                )
            )
        return results

    @staticmethod
    def _validate_depth_map(depth_map_m: np.ndarray) -> None:
        validate_ndarray(
            depth_map_m,
            expected_shape=DEPTH_SHAPE_HW,
            expected_dtype=np.float32,
            name="depth_map",
        )

    @staticmethod
    def _validate_detections(detections: np.ndarray) -> None:
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError("expected detections with shape [N, 6]")
        if detections.dtype != np.float32:
            raise ValueError("expected detections dtype float32")

    @staticmethod
    def _roi_depth(depth_map_m: np.ndarray, x: float, y: float, w: float, h: float) -> float | None:
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(depth_map_m.shape[1], int(round(x + w)))
        y1 = min(depth_map_m.shape[0], int(round(y + h)))
        if x0 >= x1 or y0 >= y1:
            return None

        roi = depth_map_m[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _roi_contour(self, depth_map_m: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(depth_map_m.shape[1], int(round(x + w)))
        y1 = min(depth_map_m.shape[0], int(round(y + h)))
        if x0 >= x1 or y0 >= y1:
            return np.zeros((0, 2), dtype=np.float32)

        contour = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )
        return contour

    def _contour_points_to_3d(self, contour_xy: np.ndarray, depth_map_m: np.ndarray) -> np.ndarray:
        if contour_xy.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        points_xyz: list[list[float]] = []
        for pixel_x, pixel_y in contour_xy:
            x_px = int(np.clip(round(float(pixel_x)), 0, depth_map_m.shape[1] - 1))
            y_px = int(np.clip(round(float(pixel_y)), 0, depth_map_m.shape[0] - 1))
            depth_value = float(depth_map_m[y_px, x_px])
            if not np.isfinite(depth_value) or depth_value <= 0:
                continue
            x_m = (float(pixel_x) - self.intrinsics.cx) * depth_value / self.intrinsics.fx
            y_m = (float(pixel_y) - self.intrinsics.cy) * depth_value / self.intrinsics.fy
            points_xyz.append([x_m, y_m, depth_value])

        if not points_xyz:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(points_xyz, dtype=np.float32)
