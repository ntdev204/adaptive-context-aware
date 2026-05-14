"""Phase 3 behavior and decision modules.

Exports are resolved lazily so packages that only need lightweight modules do
not import ``torch`` during pytest collection.
"""

from importlib import import_module

__all__ = [
    "AnomalyDetection",
    "AnomalyDetector",
    "BehaviorDecisionPipeline",
    "BehaviorDecisionResult",
    "IntentPrediction",
    "IntentPredictor",
    "NavigationCommand",
    "NavigationCommander",
    "NavigationMode",
    "RobotGoal",
]

_EXPORT_MAP = {
    "AnomalyDetection": ".anomaly_detector",
    "AnomalyDetector": ".anomaly_detector",
    "BehaviorDecisionPipeline": ".behavior_pipeline",
    "BehaviorDecisionResult": ".behavior_pipeline",
    "IntentPrediction": ".intent_predictor",
    "IntentPredictor": ".intent_predictor",
    "NavigationCommand": ".nav_commander",
    "NavigationCommander": ".nav_commander",
    "NavigationMode": ".nav_commander",
    "RobotGoal": ".nav_commander",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
