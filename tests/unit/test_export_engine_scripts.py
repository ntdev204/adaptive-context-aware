from __future__ import annotations

import sys
from types import ModuleType

import pytest

from scripts import bootstrap_engine, export_engine


def test_export_engine_prefers_sibling_engine_artifact_over_returned_non_engine_file(tmp_path) -> None:
    # Simulate Ultralytics returning a non-.engine path but leaving a sibling .engine
    intermediate_path = tmp_path / "best.pt"
    sibling_engine = tmp_path / "best.engine"
    target_engine = tmp_path / "final.engine"
    intermediate_path.write_bytes(b"weights")
    sibling_engine.write_bytes(b"engine")

    built_path = export_engine._finalize_exported_engine(
        intermediate_path,
        engine_path=target_engine,
        fp16=True,
        workspace_gb=2,
    )

    assert built_path == target_engine
    assert target_engine.read_bytes() == b"engine"
    assert not sibling_engine.exists()


def test_export_engine_rejects_non_engine_artifact_without_sibling_engine(tmp_path) -> None:
    # No sibling .engine exists — must raise rather than silently consume wrong artifact
    non_engine_path = tmp_path / "best.pt"
    non_engine_path.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="did not leave a sibling .engine artifact"):
        export_engine._finalize_exported_engine(
            non_engine_path,
            engine_path=tmp_path / "best.engine",
            fp16=True,
            workspace_gb=2,
        )


def test_bootstrap_engine_rebuilds_invalid_cached_engine_using_direct_engine_export(tmp_path, monkeypatch) -> None:
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    engine_output = engine_dir / "best.engine"
    engine_output.write_bytes(b"stale-engine")
    engine_output.with_suffix(".json").write_text("{}", encoding="utf-8")
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights")
    exported_engine = tmp_path / "best.engine"

    class FakeYOLO:
        def __init__(self, model_ref: str) -> None:
            assert model_ref == str(model_path)

        def export(self, **kwargs):
            del kwargs
            exported_engine.write_bytes(b"fresh-engine")
            return str(exported_engine)

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    monkeypatch.setattr(
        bootstrap_engine,
        "_is_valid_tensorrt_engine",
        lambda path: path.exists() and path.read_bytes() == b"fresh-engine",
    )

    result = bootstrap_engine.ensure_engine(
        engine_dir,
        model_path=model_path,
        engine_output=engine_output,
    )

    assert result == engine_output
    assert engine_output.read_bytes() == b"fresh-engine"
