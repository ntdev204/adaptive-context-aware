"""Perception layer modules.

This package uses lazy exports so lightweight imports like
``src.perception.sensor_fusion`` do not eagerly import optional runtime
dependencies such as ``cv2`` during test collection.
"""

from importlib import import_module

__all__ = [
    "CameraIntrinsics",
    "DepthBoundingBox3D",
    "DepthProcessor",
    "DetectorConfig",
    "DetectorResult",
    "EgoMotionState",
    "EntityFeatureExtractor",
    "EntityFeatures",
    "FusedEntity",
    "IMUFusion",
    "InputMode",
    "InputSource",
    "MultiObjectTracker",
    "PerceptionPipeline",
    "PerceptionPipelineReport",
    "PersonDetector",
    "SensorFusion",
]

_EXPORT_MAP = {
    "CameraIntrinsics": ".depth_proc",
    "DepthBoundingBox3D": ".depth_proc",
    "DepthProcessor": ".depth_proc",
    "DetectorConfig": ".detector",
    "DetectorResult": ".detector",
    "PersonDetector": ".detector",
    "EntityFeatureExtractor": ".feature_extractor",
    "EntityFeatures": ".feature_extractor",
    "EgoMotionState": ".imu_fusion",
    "IMUFusion": ".imu_fusion",
    "InputMode": ".input_source",
    "InputSource": ".input_source",
    "PerceptionPipeline": ".pipeline",
    "PerceptionPipelineReport": ".pipeline",
    "FusedEntity": ".sensor_fusion",
    "SensorFusion": ".sensor_fusion",
    "MultiObjectTracker": ".tracker",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
