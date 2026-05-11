from __future__ import annotations

import numpy as np

from src.perception.feature_extractor import EntityFeatureExtractor
from src.perception.sensor_fusion import FusedEntity


def _entity(track_id: int, x: float, y: float, z: float) -> FusedEntity:
    return FusedEntity(
        track_id=track_id,
        bbox_xywh=np.array([100.0, 80.0, 40.0, 120.0], dtype=np.float32),
        position_3d=np.array([x, y, z], dtype=np.float32),
        velocity_3d=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        heading_rad=0.3,
        confidence=0.9,
        nearest_obstacle_distance_m=1.5,
        nearest_obstacle_centroid_xy=np.array([x + 0.1, y + 0.1], dtype=np.float32),
        ego_velocity_xyz_mps=np.array([0.1, 0.0, 0.0], dtype=np.float32),
    )


def test_feature_extractor_output_contract() -> None:
    extractor = EntityFeatureExtractor()
    features = extractor.extract(_entity(track_id=1, x=1.0, y=0.5, z=2.0))

    assert features.track_id == 1
    assert features.appearance_embedding.shape == (128,)
    assert features.motion_features.shape == (4,)


def test_feature_extractor_same_entity_has_high_similarity() -> None:
    extractor = EntityFeatureExtractor()
    entity_a = _entity(track_id=1, x=1.0, y=0.5, z=2.0)
    entity_b = _entity(track_id=2, x=1.02, y=0.52, z=2.01)

    features_a = extractor.extract(entity_a)
    features_b = extractor.extract(entity_b)

    similarity = extractor.similarity(features_a, features_b)
    assert similarity > 0.95


def test_feature_extractor_different_entity_has_lower_similarity() -> None:
    extractor = EntityFeatureExtractor()
    entity_a = _entity(track_id=1, x=1.0, y=0.5, z=2.0)
    entity_b = _entity(track_id=2, x=4.0, y=-2.0, z=6.0)
    entity_b.velocity_3d = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    entity_b.heading_rad = -1.2
    entity_b.nearest_obstacle_distance_m = 8.0

    features_a = extractor.extract(entity_a)
    features_b = extractor.extract(entity_b)

    similarity = extractor.similarity(features_a, features_b)
    assert similarity < 0.95
