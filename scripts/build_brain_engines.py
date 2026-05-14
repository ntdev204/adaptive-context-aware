from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from models import AttentionPathway, ComplexityEstimatorNet, GatedFusion, GraphAttentionPathway, GruPathway, TcnPathway
from models.rl_policy import RLPolicyNet


@dataclass(frozen=True, slots=True)
class BrainEngineSpec:
    name: str
    model: nn.Module
    inputs: tuple[torch.Tensor, ...]
    output_path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    expected_output_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BrainEngineBuildResult:
    name: str
    output_path: Path
    pytorch_output_shape: tuple[int, ...]
    engine_exists: bool


class EngineBuilder(Protocol):
    def __call__(self, spec: BrainEngineSpec, *, fp16: bool = True) -> Path:
        """Build one TensorRT engine and return its path."""


class FusionEngineWrapper(nn.Module):
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


def build_engine_specs(output_dir: Path = Path("models/engines")) -> list[BrainEngineSpec]:
    return [
        BrainEngineSpec(
            name="estimator",
            model=ComplexityEstimatorNet().eval(),
            inputs=(torch.zeros(1, 36, dtype=torch.float32),),
            output_path=output_dir / "estimator.engine",
            input_names=("complexity_features",),
            output_names=("logits",),
            expected_output_shape=(1, 4),
        ),
        BrainEngineSpec(
            name="gru",
            model=GruPathway().eval(),
            inputs=(torch.zeros(2, 8, 128, dtype=torch.float32),),
            output_path=output_dir / "gru_pathway.engine",
            input_names=("sequence_features",),
            output_names=("summary",),
            expected_output_shape=(2, 64),
        ),
        BrainEngineSpec(
            name="tcn",
            model=TcnPathway().eval(),
            inputs=(torch.zeros(2, 128, 8, dtype=torch.float32),),
            output_path=output_dir / "tcn_pathway.engine",
            input_names=("sequence_features",),
            output_names=("summary",),
            expected_output_shape=(2, 64),
        ),
        BrainEngineSpec(
            name="attention",
            model=AttentionPathway().eval(),
            inputs=(torch.zeros(2, 5, 128, dtype=torch.float32),),
            output_path=output_dir / "attention_pathway.engine",
            input_names=("entity_features",),
            output_names=("context",),
            expected_output_shape=(2, 128),
        ),
        BrainEngineSpec(
            name="gnn",
            model=GraphAttentionPathway().eval(),
            inputs=(
                torch.zeros(2, 5, 128, dtype=torch.float32),
                torch.ones(2, 5, 5, dtype=torch.float32),
            ),
            output_path=output_dir / "gnn_pathway.engine",
            input_names=("entity_features", "adjacency"),
            output_names=("graph_context",),
            expected_output_shape=(2, 256),
        ),
        BrainEngineSpec(
            name="fusion",
            model=FusionEngineWrapper(GatedFusion()).eval(),
            inputs=(
                torch.zeros(2, 512, dtype=torch.float32),
                torch.ones(2, 4, dtype=torch.float32),
            ),
            output_path=output_dir / "gated_fusion.engine",
            input_names=("pathway_outputs", "active_mask"),
            output_names=("fused",),
            expected_output_shape=(2, 256),
        ),
        BrainEngineSpec(
            name="rl_policy",
            model=RLPolicyNet().eval(),
            inputs=(torch.zeros(1, 39, dtype=torch.float32),),
            output_path=output_dir / "rl_policy.engine",
            input_names=("router_state",),
            output_names=("action_logits",),
            expected_output_shape=(1, 4),
        ),
    ]


def build_brain_engines(
    output_dir: Path = Path("models/engines"),
    *,
    builder: EngineBuilder | None = None,
    fp16: bool = True,
) -> list[BrainEngineBuildResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_builder = builder or torch_tensorrt_engine_builder
    results = []
    for spec in build_engine_specs(output_dir):
        pytorch_output = run_pytorch_shape_check(spec)
        engine_builder(spec, fp16=fp16)
        _write_engine_metadata(spec, fp16=fp16)
        results.append(
            BrainEngineBuildResult(
                name=spec.name,
                output_path=spec.output_path,
                pytorch_output_shape=tuple(pytorch_output.shape),
                engine_exists=spec.output_path.exists(),
            )
        )
    return results


def run_pytorch_shape_check(spec: BrainEngineSpec) -> torch.Tensor:
    spec.model.eval()
    with torch.no_grad():
        output = spec.model(*spec.inputs)
    if tuple(output.shape) != spec.expected_output_shape:
        raise ValueError(f"{spec.name} PyTorch output shape {tuple(output.shape)} != {spec.expected_output_shape}")
    return output


def torch_tensorrt_engine_builder(spec: BrainEngineSpec, *, fp16: bool = True) -> Path:
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError("torch-tensorrt is required to build Phase 2 .engine files on Jetson") from exc

    compile_kwargs = {
        "inputs": [tuple(tensor.shape) for tensor in spec.inputs],
        "enabled_precisions": {torch.float16} if fp16 else {torch.float32},
    }
    compiled = torch_tensorrt.compile(spec.model, **compile_kwargs)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, str(spec.output_path))
    return spec.output_path


def _write_engine_metadata(spec: BrainEngineSpec, *, fp16: bool) -> None:
    metadata = {
        "name": spec.name,
        "engine_path": str(spec.output_path),
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
        "expected_output_shape": list(spec.expected_output_shape),
        "precision": "fp16" if fp16 else "fp32",
        "format": "TensorRT engine",
    }
    spec.output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 TensorRT .engine files on Jetson.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/engines"))
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    results = build_brain_engines(output_dir=args.output_dir, fp16=not args.fp32)
    for result in results:
        print(
            f"{result.name}: path={result.output_path} "
            f"shape={result.pytorch_output_shape} engine_exists={result.engine_exists}"
        )


if __name__ == "__main__":
    main()
