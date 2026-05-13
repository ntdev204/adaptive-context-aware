from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.constants import FRAME_HEIGHT, FRAME_WIDTH

from .sensor_fusion import FusedEntity


@dataclass(slots=True)
class EntityFeatures:
    track_id: int
    appearance_embedding: np.ndarray
    motion_features: np.ndarray


class EntityFeatureExtractor:
    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim

    def extract(self, entity: FusedEntity) -> EntityFeatures:
        base_vector = np.array(
            [
                entity.bbox_xywh[0] / FRAME_WIDTH,
                entity.bbox_xywh[1] / FRAME_HEIGHT,
                entity.bbox_xywh[2] / FRAME_WIDTH,
                entity.bbox_xywh[3] / FRAME_HEIGHT,
                entity.position_3d[0],
                entity.position_3d[1],
                entity.position_3d[2],
                entity.velocity_3d[0],
                entity.velocity_3d[1],
                entity.velocity_3d[2],
                entity.heading_rad,
                entity.confidence,
                entity.nearest_obstacle_distance_m or -1.0,
                entity.ego_velocity_xyz_mps[0],
                entity.ego_velocity_xyz_mps[1],
                entity.ego_velocity_xyz_mps[2],
            ],
            dtype=np.float32,
        )

        embedding = self._project_to_embedding(base_vector)
        motion_features = np.array(
            [
                float(np.linalg.norm(entity.velocity_3d)),
                entity.heading_rad,
                entity.position_3d[2],
                entity.nearest_obstacle_distance_m or -1.0,
            ],
            dtype=np.float32,
        )
        return EntityFeatures(
            track_id=entity.track_id,
            appearance_embedding=embedding,
            motion_features=motion_features,
        )

    def similarity(self, left: EntityFeatures, right: EntityFeatures) -> float:
        a = left.appearance_embedding
        b = right.appearance_embedding
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _project_to_embedding(self, base_vector: np.ndarray) -> np.ndarray:
        repeats = int(np.ceil(self.embedding_dim / base_vector.shape[0]))
        tiled = np.tile(base_vector, repeats)[: self.embedding_dim]
        norm = np.linalg.norm(tiled)
        if norm > 0:
            tiled = tiled / norm
        return tiled.astype(np.float32)
