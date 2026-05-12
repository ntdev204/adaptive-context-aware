from __future__ import annotations

import numpy as np
import pytest

from src.perception.lidar_proc import LidarProcessor


def test_lidar_processor_clusters_mock_scan() -> None:
    processor = LidarProcessor(distance_jump_threshold_m=0.2, min_points_per_cluster=3)
    scan = np.array(
        [
            [0.00, 2.0],
            [0.03, 2.0],
            [0.06, 2.0],
            [0.50, 4.0],
            [0.53, 4.0],
            [0.56, 4.0],
        ],
        dtype=np.float32,
    )

    clusters = processor.cluster_scan(scan)

    assert len(clusters) == 2
    assert clusters[0].mean_range_m == pytest.approx(2.0, abs=0.2)
    assert clusters[1].mean_range_m == pytest.approx(4.0, abs=0.2)


def test_lidar_processor_filters_invalid_points() -> None:
    processor = LidarProcessor(distance_jump_threshold_m=1.0)
    scan = np.array(
        [
            [0.00, 2.0],
            [0.10, -1.0],
            [0.20, 999.0],
            [0.30, np.nan],
            [0.40, 2.1],
            [0.50, 2.2],
        ],
        dtype=np.float32,
    )

    clusters = processor.cluster_scan(scan)

    assert len(clusters) == 1
    assert clusters[0].points_xy.shape[0] == 3


def test_lidar_processor_rejects_wrong_shape() -> None:
    processor = LidarProcessor()
    scan = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="shape \\[N, 2\\]"):
        processor.cluster_scan(scan)
