from __future__ import annotations

import torch

from scripts.export_brain_onnx import build_export_specs, export_brain_onnx, run_pytorch_shape_check


def test_export_specs_cover_phase_2_brain_models(tmp_path) -> None:
    specs = build_export_specs(tmp_path)

    assert [spec.name for spec in specs] == ["estimator", "gru", "tcn", "attention", "gnn", "fusion"]
    assert [spec.output_path.name for spec in specs] == [
        "estimator.onnx",
        "gru_pathway.onnx",
        "tcn_pathway.onnx",
        "attention_pathway.onnx",
        "gnn_pathway.onnx",
        "gated_fusion.onnx",
    ]


def test_export_specs_match_pytorch_output_shapes(tmp_path) -> None:
    for spec in build_export_specs(tmp_path):
        output = run_pytorch_shape_check(spec)
        assert tuple(output.shape) == spec.expected_output_shape


def test_export_brain_onnx_writes_all_files_with_mock_export(tmp_path, monkeypatch) -> None:
    def fake_export(
        model: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output_path,
        **_: object,
    ) -> None:
        del model, inputs
        output_path.write_bytes(b"fake-onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    results = export_brain_onnx(tmp_path, run_checker=False, run_runtime=False)

    assert len(results) == 6
    assert all(result.output_path.exists() for result in results)
    assert all(result.checker_passed is None for result in results)
