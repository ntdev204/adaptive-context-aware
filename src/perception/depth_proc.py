from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    class_id: float


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

        results: list[DepthBoundingBox3D] = []
        for x, y, w, h, conf, cls in detections:
            depth_value = self._roi_depth(depth_map_m, x, y, w, h)
            if depth_value is None:
                continue
            center_x = float(x + w / 2.0)
            center_y = float(y + h / 2.0)
            x_m = (center_x - self.intrinsics.cx) * depth_value / self.intrinsics.fx
            y_m = (center_y - self.intrinsics.cy) * depth_value / self.intrinsics.fy
            width_m = float(w) * depth_value / self.intrinsics.fx
            height_m = float(h) * depth_value / self.intrinsics.fy
            results.append(
                DepthBoundingBox3D(
                    x_m=x_m,
                    y_m=y_m,
                    z_m=depth_value,
                    width_m=width_m,
                    height_m=height_m,
                    confidence=float(conf),
                    class_id=float(cls),
                )
            )
        return results

    @staticmethod
    def _validate_depth_map(depth_map_m: np.ndarray) -> None:
        if depth_map_m.shape != (480, 640):
            raise ValueError("expected depth map shape (480, 640)")
        if depth_map_m.dtype != np.float32:
            raise ValueError("expected depth map dtype float32")

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
