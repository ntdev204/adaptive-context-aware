from __future__ import annotations

from pathlib import Path

from scripts.export_engine import _copy_engine_artifact, _resolve_outputs


def test_resolve_outputs_writes_engines_to_output_dir(tmp_path: Path) -> None:
    model_path = tmp_path / "fine_tuning" / "best.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"weights")

    output_dir = tmp_path / "engines"
    mapping = _resolve_outputs([model_path], output_dir)

    assert mapping == {model_path: output_dir / "best.engine"}


def test_copy_engine_artifact_syncs_to_codebase_dir(tmp_path: Path) -> None:
    engine_path = tmp_path / "container" / "best.engine"
    engine_path.parent.mkdir()
    engine_path.write_bytes(b"fake-engine")

    codebase_path = tmp_path / "codebase" / "models" / "engines" / "best.engine"
    _copy_engine_artifact(engine_path, codebase_path)

    assert codebase_path.read_bytes() == b"fake-engine"
