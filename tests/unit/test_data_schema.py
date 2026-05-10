from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from src.utils.enums import Activity, IntentDirection, SceneContext


def test_annotation_json_schema() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schemas" / "annotation_schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    sample = {
        "frame_id": 0,
        "timestamp": 1715000000000000,
        "persons": [
            {
                "bbox": [120, 80, 50, 120],
                "track_id": 1,
                "activity": "WALKING",
                "intent_direction": "NORTH",
                "trajectory_pred": [[130, 75], [140, 70]],
                "is_anomaly": False,
                "confidence": 0.92,
            }
        ],
        "scene": {
            "context": "CORRIDOR",
            "crowd_density": 0.15,
            "motion_entropy": 0.22,
            "anomaly_flag": False,
        },
    }
    validate(instance=sample, schema=schema)


def test_enums_match_spec() -> None:
    assert Activity.FIGHTING.value == "FIGHTING"
    assert SceneContext.GATE_AREA.value == "GATE_AREA"
    assert IntentDirection.STATIONARY.value == "STATIONARY"
