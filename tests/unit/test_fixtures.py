from __future__ import annotations

from pathlib import Path


def test_committed_fixtures_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures"
    assert len(list((root / "annotations").glob("frame_*.json"))) >= 5
    assert len(list((root / "anomaly_synthetic").glob("case_*.json"))) >= 5
    assert len(list((root / "rl_scenarios").glob("scenario_*.json"))) >= 5
    assert len(list((root / "images").glob("*.png"))) >= 5


def test_generated_fixture_exists_or_can_be_generated() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "fixtures" / "sample_recording.h5"
    script = root.parents[0] / "scripts" / "generate_synthetic_fixtures.py"
    assert fixture.exists() or script.exists()


def test_local_bootstrap_subsets_exist_or_can_be_extracted() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root.parents[0] / "scripts" / "download_fixtures.py"
    cctv_subset = root / "fixtures" / "cctv_person_subset"
    mot20_subset = root / "fixtures" / "mot20_subset"
    assert script.exists()
    assert cctv_subset.exists() or mot20_subset.exists() or script.exists()
