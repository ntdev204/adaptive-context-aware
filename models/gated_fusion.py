from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Gated fusion layer for active Phase 2 reasoning pathways."""

    DEFAULT_PATHWAY_DIMS = {
        "gru": 64,
        "tcn": 64,
        "attention": 128,
        "gnn": 256,
    }

    def __init__(
        self,
        pathway_dims: Mapping[str, int] | None = None,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pathway_dims = dict(pathway_dims or self.DEFAULT_PATHWAY_DIMS)
        self.pathway_order = tuple(self.pathway_dims)
        self.output_dim = output_dim
        self.input_dim = sum(self.pathway_dims.values())
        self._offsets = self._build_offsets()

        self.gate_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.pathway_order)),
        )
        self.projection = nn.Linear(self.input_dim, output_dim)

    def forward(
        self,
        pathway_outputs: Mapping[str, torch.Tensor] | torch.Tensor,
        active_pathways: Sequence[str] | None = None,
    ) -> torch.Tensor:
        fused, _ = self.forward_with_gates(pathway_outputs, active_pathways=active_pathways)
        return fused

    def forward_with_gates(
        self,
        pathway_outputs: Mapping[str, torch.Tensor] | torch.Tensor,
        active_pathways: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded_input, active_indices = self._prepare_input(pathway_outputs, active_pathways)
        gate_logits = self.gate_network(padded_input)

        active_mask = torch.zeros_like(gate_logits, dtype=torch.bool)
        active_mask[:, active_indices] = True
        gate_logits = gate_logits.masked_fill(~active_mask, torch.finfo(gate_logits.dtype).min)
        all_gates = torch.softmax(gate_logits, dim=-1)

        fused_input = self._apply_gates(padded_input, all_gates)
        return self.projection(fused_input), all_gates[:, active_indices]

    def _prepare_input(
        self,
        pathway_outputs: Mapping[str, torch.Tensor] | torch.Tensor,
        active_pathways: Sequence[str] | None,
    ) -> tuple[torch.Tensor, list[int]]:
        if isinstance(pathway_outputs, Mapping):
            if active_pathways is not None:
                raise ValueError("active_pathways is only valid with concatenated tensor input")
            return self._prepare_mapping_input(pathway_outputs)
        if active_pathways is None:
            raise ValueError("active_pathways is required for concatenated tensor input")
        return self._prepare_concatenated_input(pathway_outputs, active_pathways)

    def _prepare_mapping_input(self, pathway_outputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        unknown_pathways = sorted(set(pathway_outputs) - set(self.pathway_dims))
        if unknown_pathways:
            raise ValueError(f"unknown pathway output(s): {', '.join(unknown_pathways)}")
        if not pathway_outputs:
            raise ValueError("gated fusion requires at least one active pathway")

        sample = next(iter(pathway_outputs.values()))
        batch_size = self._validate_tensor(sample, expected_dim=sample.shape[-1])
        padded_input = sample.new_zeros(batch_size, self.input_dim)
        active_indices: list[int] = []

        for index, name in enumerate(self.pathway_order):
            if name not in pathway_outputs:
                continue
            values = pathway_outputs[name]
            batch_size = self._validate_tensor(values, expected_dim=self.pathway_dims[name], batch_size=batch_size)
            start, end = self._offsets[name]
            padded_input[:, start:end] = values
            active_indices.append(index)

        return padded_input, active_indices

    def _prepare_concatenated_input(
        self,
        pathway_outputs: torch.Tensor,
        active_pathways: Sequence[str],
    ) -> tuple[torch.Tensor, list[int]]:
        if isinstance(active_pathways, str):
            raise ValueError("active_pathways must be a sequence of pathway names")
        active_names = list(active_pathways)
        if not active_names:
            raise ValueError("gated fusion requires at least one active pathway")
        if len(set(active_names)) != len(active_names):
            raise ValueError("active_pathways must not contain duplicates")

        unknown_pathways = sorted(set(active_names) - set(self.pathway_dims))
        if unknown_pathways:
            raise ValueError(f"unknown pathway output(s): {', '.join(unknown_pathways)}")

        expected_dim = sum(self.pathway_dims[name] for name in active_names)
        batch_size = self._validate_tensor(pathway_outputs, expected_dim=expected_dim)
        padded_input = pathway_outputs.new_zeros(batch_size, self.input_dim)
        active_indices: list[int] = []

        input_offset = 0
        for name in active_names:
            dim = self.pathway_dims[name]
            start, end = self._offsets[name]
            padded_input[:, start:end] = pathway_outputs[:, input_offset : input_offset + dim]
            active_indices.append(self.pathway_order.index(name))
            input_offset += dim

        return padded_input, active_indices

    def _apply_gates(self, padded_input: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        weighted_segments = []
        for index, name in enumerate(self.pathway_order):
            start, end = self._offsets[name]
            weighted_segments.append(padded_input[:, start:end] * gates[:, index : index + 1])
        return torch.cat(weighted_segments, dim=-1)

    @staticmethod
    def _validate_tensor(
        values: torch.Tensor,
        expected_dim: int,
        batch_size: int | None = None,
    ) -> int:
        if values.ndim != 2 or values.shape[-1] != expected_dim:
            raise ValueError(f"expected pathway output shape [B, {expected_dim}]")
        if batch_size is not None and values.shape[0] != batch_size:
            raise ValueError("all pathway outputs must share the same batch size")
        return values.shape[0]

    def _build_offsets(self) -> dict[str, tuple[int, int]]:
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, dim in self.pathway_dims.items():
            offsets[name] = (cursor, cursor + dim)
            cursor += dim
        return offsets
