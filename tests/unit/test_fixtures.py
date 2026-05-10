from __future__ import annotations

from pathlib import Path


def test_committed_fixtures_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    assert (root / "annotations" / "frame_000.json").exists()
    assert (root / "anomaly_synthetic" / "case_01.json").exists()
    assert (root / "rl_scenarios" / "scenario_01.json").exists()


def test_generated_fixture_exists_or_can_be_generated() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "fixtures" / "sample_recording.h5"
    script = root.parents[0] / "scripts" / "generate_synthetic_fixtures.py"
    assert fixture.exists() or script.exists()
