from __future__ import annotations

from dataclasses import dataclass

from src.router.adaptive_router import ReasoningPathway


@dataclass(frozen=True, slots=True)
class SafetyMetrics:
    fps: float
    total_latency_ms: float
    detector_latency_ms: float
    fusion_latency_ms: float
    gpu_util_pct: float
    cpu_util_pct: float
    ram_used_mb: float
    pathway_selection: dict[ReasoningPathway, int]

    def as_json_payload(self) -> dict[str, object]:
        return {
            "fps": self.fps,
            "total_latency_ms": self.total_latency_ms,
            "detector_latency_ms": self.detector_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "gpu_util_pct": self.gpu_util_pct,
            "cpu_util_pct": self.cpu_util_pct,
            "ram_used_mb": self.ram_used_mb,
            "pathway_selection": {pathway.value: count for pathway, count in self.pathway_selection.items()},
        }
