from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from src.utils.enums import Activity, IntentDirection, SceneContext


def test_annotation_json_schema_accepts_committed_fixtures() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schemas" / "annotation_schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "annotations"
    fixture_paths = sorted(fixtures_dir.glob("frame_*.json"))
    assert fixture_paths, "expected committed annotation fixtures"

    for fixture_path in fixture_paths:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        validate(instance=payload, schema=schema)


def test_annotation_json_schema_rejects_invalid_activity_enum() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schemas" / "annotation_schema.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    invalid = {
        "frame_id": 0,
        "timestamp": 1715000000000000,
        "persons": [
            {
                "bbox": [120, 80, 50, 120],
                "track_id": 1,
                "activity": "JUMPING",
            }
        ],
        "scene": {
            "context": "CORRIDOR",
            "crowd_density": 0.15,
        },
    }

    with pytest.raises(ValidationError):
        validate(instance=invalid, schema=schema)


def test_enums_match_spec() -> None:
    assert Activity.FIGHTING.value == "FIGHTING"
    assert Activity.LOITERING.value == "LOITERING"
    assert SceneContext.GATE_AREA.value == "GATE_AREA"
    assert SceneContext.OPEN_SPACE.value == "OPEN_SPACE"
    assert IntentDirection.STATIONARY.value == "STATIONARY"
    assert IntentDirection.NW.value == "NW"
