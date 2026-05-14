from __future__ import annotations

from scripts.build_brain_engines import build_brain_engines, build_engine_specs, run_pytorch_shape_check


def test_engine_specs_cover_phase_2_brain_models(tmp_path) -> None:
    specs = build_engine_specs(tmp_path)

    assert [spec.name for spec in specs] == ["estimator", "gru", "tcn", "attention", "gnn", "fusion", "rl_policy"]
    assert [spec.output_path.name for spec in specs] == [
        "estimator.engine",
        "gru_pathway.engine",
        "tcn_pathway.engine",
        "attention_pathway.engine",
        "gnn_pathway.engine",
        "gated_fusion.engine",
        "rl_policy.engine",
    ]


def test_engine_specs_match_pytorch_output_shapes(tmp_path) -> None:
    for spec in build_engine_specs(tmp_path):
        output = run_pytorch_shape_check(spec)
        assert tuple(output.shape) == spec.expected_output_shape


def test_build_brain_engines_writes_all_files_with_mock_builder(tmp_path) -> None:
    def fake_builder(spec, *, fp16: bool = True):
        assert fp16
        spec.output_path.write_bytes(b"fake-engine")
        return spec.output_path

    results = build_brain_engines(tmp_path, builder=fake_builder)

    assert len(results) == 7
    assert all(result.output_path.exists() for result in results)
    assert all(result.output_path.with_suffix(".json").exists() for result in results)
    assert all(result.engine_exists for result in results)
