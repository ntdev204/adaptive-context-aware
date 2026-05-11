from __future__ import annotations

import numpy as np
import pytest

from src.perception.depth_proc import CameraIntrinsics, DepthProcessor


def test_depth_processor_projects_detection_to_3d() -> None:
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    processor = DepthProcessor(intrinsics)
    depth_map = np.full((480, 640), 2.0, dtype=np.float32)
    detections = np.array([[300.0, 220.0, 40.0, 80.0, 0.95, 0.0]], dtype=np.float32)

    results = processor.detections_to_3d(depth_map, detections)

    assert len(results) == 1
    result = results[0]
    assert result.x_m == pytest.approx(0.0, abs=0.3)
    assert result.y_m == pytest.approx(0.1, abs=0.3)
    assert result.z_m == pytest.approx(2.0, abs=0.3)
    assert result.width_m == pytest.approx(0.2, abs=0.3)
    assert result.height_m == pytest.approx(0.4, abs=0.3)


def test_depth_processor_ignores_invalid_depth_roi() -> None:
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    processor = DepthProcessor(intrinsics)
    depth_map = np.zeros((480, 640), dtype=np.float32)
    detections = np.array([[300.0, 220.0, 40.0, 80.0, 0.95, 0.0]], dtype=np.float32)

    results = processor.detections_to_3d(depth_map, detections)

    assert results == []


def test_depth_processor_rejects_wrong_detection_contract() -> None:
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    processor = DepthProcessor(intrinsics)
    depth_map = np.full((480, 640), 2.0, dtype=np.float32)
    detections = np.array([[300.0, 220.0, 40.0, 80.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="shape \\[N, 6\\]"):
        processor.detections_to_3d(depth_map, detections)
