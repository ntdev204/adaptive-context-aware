from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import AttentionPathway, ComplexityEstimatorNet, GatedFusion, GraphAttentionPathway, GruPathway, TcnPathway


@dataclass(frozen=True, slots=True)
class BrainExportSpec:
    name: str
    model: nn.Module
    inputs: tuple[torch.Tensor, ...]
    output_path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_axes: dict[str, dict[int, str]]
    expected_output_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BrainExportResult:
    name: str
    output_path: Path
    pytorch_output_shape: tuple[int, ...]
    checker_passed: bool | None
    runtime_max_abs_diff: float | None


class FusionExportWrapper(nn.Module):
    def __init__(self, fusion: GatedFusion) -> None:
        super().__init__()
        self.fusion = fusion

    def forward(self, pathway_outputs: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        gate_logits = self.fusion.gate_network(pathway_outputs)
        masked_logits = gate_logits.masked_fill(active_mask <= 0.0, torch.finfo(gate_logits.dtype).min)
        gates = torch.softmax(masked_logits, dim=-1)

        weighted_segments = []
        offset = 0
        for index, pathway in enumerate(self.fusion.pathway_order):
            dim = self.fusion.pathway_dims[pathway]
            weighted_segments.append(pathway_outputs[:, offset : offset + dim] * gates[:, index : index + 1])
            offset += dim
        return self.fusion.projection(torch.cat(weighted_segments, dim=-1))


def build_export_specs(output_dir: Path = Path("models/onnx")) -> list[BrainExportSpec]:
    return [
        BrainExportSpec(
            name="estimator",
            model=ComplexityEstimatorNet().eval(),
            inputs=(torch.zeros(1, 36, dtype=torch.float32),),
            output_path=output_dir / "estimator.onnx",
            input_names=("complexity_features",),
            output_names=("logits",),
            dynamic_axes={"complexity_features": {0: "batch"}, "logits": {0: "batch"}},
            expected_output_shape=(1, 4),
        ),
        BrainExportSpec(
            name="gru",
            model=GruPathway().eval(),
            inputs=(torch.zeros(2, 8, 128, dtype=torch.float32),),
            output_path=output_dir / "gru_pathway.onnx",
            input_names=("sequence_features",),
            output_names=("summary",),
            dynamic_axes={"sequence_features": {0: "batch", 1: "time"}, "summary": {0: "batch"}},
            expected_output_shape=(2, 64),
        ),
        BrainExportSpec(
            name="tcn",
            model=TcnPathway().eval(),
            inputs=(torch.zeros(2, 128, 8, dtype=torch.float32),),
            output_path=output_dir / "tcn_pathway.onnx",
            input_names=("sequence_features",),
            output_names=("summary",),
            dynamic_axes={"sequence_features": {0: "batch", 2: "time"}, "summary": {0: "batch"}},
            expected_output_shape=(2, 64),
        ),
        BrainExportSpec(
            name="attention",
            model=AttentionPathway().eval(),
            inputs=(torch.zeros(2, 5, 128, dtype=torch.float32),),
            output_path=output_dir / "attention_pathway.onnx",
            input_names=("entity_features",),
            output_names=("context",),
            dynamic_axes={"entity_features": {0: "batch", 1: "entities"}, "context": {0: "batch"}},
            expected_output_shape=(2, 128),
        ),
        BrainExportSpec(
            name="gnn",
            model=GraphAttentionPathway().eval(),
            inputs=(
                torch.zeros(2, 5, 128, dtype=torch.float32),
                torch.ones(2, 5, 5, dtype=torch.float32),
            ),
            output_path=output_dir / "gnn_pathway.onnx",
            input_names=("entity_features", "adjacency"),
            output_names=("graph_context",),
            dynamic_axes={
                "entity_features": {0: "batch", 1: "entities"},
                "adjacency": {0: "batch", 1: "entities", 2: "entities"},
                "graph_context": {0: "batch"},
            },
            expected_output_shape=(2, 256),
        ),
        BrainExportSpec(
            name="fusion",
            model=FusionExportWrapper(GatedFusion()).eval(),
            inputs=(
                torch.zeros(2, 512, dtype=torch.float32),
                torch.ones(2, 4, dtype=torch.float32),
            ),
            output_path=output_dir / "gated_fusion.onnx",
            input_names=("pathway_outputs", "active_mask"),
            output_names=("fused",),
            dynamic_axes={"pathway_outputs": {0: "batch"}, "active_mask": {0: "batch"}, "fused": {0: "batch"}},
            expected_output_shape=(2, 256),
        ),
    ]


def export_brain_onnx(
    output_dir: Path = Path("models/onnx"),
    *,
    run_checker: bool = True,
    run_runtime: bool = False,
    require_optional_tools: bool = False,
) -> list[BrainExportResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in build_export_specs(output_dir):
        pytorch_output = run_pytorch_shape_check(spec)
        torch.onnx.export(
            spec.model,
            spec.inputs,
            spec.output_path,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            dynamic_axes=spec.dynamic_axes,
            opset_version=17,
        )
        checker_passed = check_onnx_model(spec.output_path, require_optional_tools) if run_checker else None
        runtime_max_abs_diff = (
            verify_onnx_runtime(spec, pytorch_output, require_optional_tools) if run_runtime else None
        )
        results.append(
            BrainExportResult(
                name=spec.name,
                output_path=spec.output_path,
                pytorch_output_shape=tuple(pytorch_output.shape),
                checker_passed=checker_passed,
                runtime_max_abs_diff=runtime_max_abs_diff,
            )
        )
    return results


def run_pytorch_shape_check(spec: BrainExportSpec) -> torch.Tensor:
    spec.model.eval()
    with torch.no_grad():
        output = spec.model(*spec.inputs)
    if tuple(output.shape) != spec.expected_output_shape:
        raise ValueError(f"{spec.name} PyTorch output shape {tuple(output.shape)} != {spec.expected_output_shape}")
    return output


def check_onnx_model(output_path: Path, require_optional_tools: bool = False) -> bool | None:
    try:
        import onnx
    except ImportError:
        if require_optional_tools:
            raise
        return None
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    return True


def verify_onnx_runtime(
    spec: BrainExportSpec,
    pytorch_output: torch.Tensor,
    require_optional_tools: bool = False,
) -> float | None:
    try:
        import onnxruntime as ort
    except ImportError:
        if require_optional_tools:
            raise
        return None
    session = ort.InferenceSession(str(spec.output_path), providers=["CPUExecutionProvider"])
    feed = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in zip(spec.input_names, spec.inputs)
    }
    onnx_output = session.run(None, feed)[0]
    max_abs_diff = float(np.max(np.abs(onnx_output - pytorch_output.detach().cpu().numpy())))
    return max_abs_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 2 brain models to ONNX.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--skip-checker", action="store_true")
    parser.add_argument("--run-runtime", action="store_true")
    parser.add_argument("--require-optional-tools", action="store_true")
    args = parser.parse_args()

    results = export_brain_onnx(
        output_dir=args.output_dir,
        run_checker=not args.skip_checker,
        run_runtime=args.run_runtime,
        require_optional_tools=args.require_optional_tools,
    )
    for result in results:
        print(
            f"{result.name}: path={result.output_path} "
            f"shape={result.pytorch_output_shape} checker={result.checker_passed} "
            f"runtime_max_abs_diff={result.runtime_max_abs_diff}"
        )


if __name__ == "__main__":
    main()
