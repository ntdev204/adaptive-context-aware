from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LidarCluster:
    points_xy: np.ndarray
    centroid_xy: np.ndarray
    mean_range_m: float
    radius_m: float


class LidarProcessor:
    def __init__(
        self,
        distance_jump_threshold_m: float = 0.35,
        min_points_per_cluster: int = 3,
        max_range_m: float = 20.0,
    ) -> None:
        self.distance_jump_threshold_m = distance_jump_threshold_m
        self.min_points_per_cluster = min_points_per_cluster
        self.max_range_m = max_range_m

    def cluster_scan(self, scan_points: np.ndarray) -> list[LidarCluster]:
        points = self._validate_and_filter(scan_points)
        if points.shape[0] == 0:
            return []

        xy_points = self._polar_to_cartesian(points)

        # Vectorized jump detection — replaces Python loop over individual points
        diffs = xy_points[1:] - xy_points[:-1]  # (N-1, 2)
        jumps = np.linalg.norm(diffs, axis=1)  # (N-1,)
        split_indices = np.flatnonzero(jumps > self.distance_jump_threshold_m) + 1

        raw_clusters = np.split(xy_points, split_indices)

        results: list[LidarCluster] = []
        for cluster in raw_clusters:
            if cluster.shape[0] < self.min_points_per_cluster:
                continue
            centroid = cluster.mean(axis=0)
            dists = np.linalg.norm(cluster - centroid, axis=1)
            results.append(
                LidarCluster(
                    points_xy=cluster,
                    centroid_xy=centroid,
                    mean_range_m=float(np.linalg.norm(cluster, axis=1).mean()),
                    radius_m=float(dists.max()),
                )
            )
        return results

    def _validate_and_filter(self, scan_points: np.ndarray) -> np.ndarray:
        if scan_points.ndim != 2 or scan_points.shape[1] != 2:
            raise ValueError("expected LiDAR scan shape [N, 2]")
        if scan_points.dtype != np.float32:
            raise ValueError("expected LiDAR scan dtype float32")

        angle = scan_points[:, 0]
        distance = scan_points[:, 1]
        valid_mask = np.isfinite(angle) & np.isfinite(distance) & (distance > 0) & (distance <= self.max_range_m)
        return scan_points[valid_mask]

    @staticmethod
    def _polar_to_cartesian(scan_points: np.ndarray) -> np.ndarray:
        angles = scan_points[:, 0]
        ranges = scan_points[:, 1]
        x = np.cos(angles) * ranges
        y = np.sin(angles) * ranges
        return np.stack((x, y), axis=1).astype(np.float32)
