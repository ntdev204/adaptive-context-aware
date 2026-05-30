from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts.export_engine import _cleanup_intermediate_exports, _copy_engine_artifact, _resolve_outputs, export_model


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


def test_cleanup_intermediate_exports_removes_onnx_side_product(tmp_path: Path) -> None:
    model_path = tmp_path / "fine_tuning" / "best.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"weights")
    onnx_path = model_path.with_suffix(".onnx")
    onnx_path.write_bytes(b"intermediate")

    _cleanup_intermediate_exports(model_path)

    assert model_path.exists()
    assert not onnx_path.exists()


def test_export_model_uses_direct_engine_export_and_removes_onnx(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "fine_tuning" / "best.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"weights")
    output_path = tmp_path / "engines" / "best.engine"
    captured_kwargs = {}

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(model_path)

        def export(self, **kwargs):
            captured_kwargs.update(kwargs)
            model_path.with_suffix(".onnx").write_bytes(b"intermediate")
            exported = model_path.with_suffix(".engine")
            exported.write_bytes(b"engine")
            return exported

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    exported = export_model(
        pt_path=model_path,
        engine_path=output_path,
        fp16=True,
        int8=True,
        dynamic=True,
        batch=8,
        data="coco.yaml",
        imgsz=[480, 640],
        workspace=4,
    )

    assert exported
    assert output_path.read_bytes() == b"engine"
    assert not model_path.with_suffix(".onnx").exists()
    assert captured_kwargs["format"] == "engine"
    assert captured_kwargs["dynamic"] is True
    assert captured_kwargs["batch"] == 8
    assert captured_kwargs["workspace"] == 4
    assert captured_kwargs["int8"] is True
    assert captured_kwargs["half"] is False
    assert captured_kwargs["data"] == "coco.yaml"
