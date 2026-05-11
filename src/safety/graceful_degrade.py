from __future__ import annotations

from dataclasses import dataclass

from src.router.adaptive_router import ReasoningPathway
from src.utils.enums import SafetyState, StatusChangeReason


@dataclass(frozen=True, slots=True)
class DegradationDecision:
    state: SafetyState
    reason: StatusChangeReason
    max_speed_scale: float
    target_fps: int
    active_pathways: tuple[ReasoningPathway, ...]
    mode: str


def decide_degradation(
    *,
    gpu_oom: bool = False,
    camera_failed: bool = False,
    gpu_temp_c: float | None = None,
) -> DegradationDecision | None:
    if gpu_oom:
        return DegradationDecision(
            state=SafetyState.DEGRADED,
            reason=StatusChangeReason.SOH_LOW,
            max_speed_scale=0.5,
            target_fps=15,
            active_pathways=(ReasoningPathway.GRU,),
            mode="gru_only",
        )
    if camera_failed:
        return DegradationDecision(
            state=SafetyState.DEGRADED,
            reason=StatusChangeReason.CAMERA_FAIL,
            max_speed_scale=0.5,
            target_fps=15,
            active_pathways=(ReasoningPathway.GRU,),
            mode="lidar_only",
        )
    if gpu_temp_c is not None and gpu_temp_c > 80.0:
        return DegradationDecision(
            state=SafetyState.DEGRADED,
            reason=StatusChangeReason.GPU_OVERHEAT,
            max_speed_scale=0.6,
            target_fps=20,
            active_pathways=(ReasoningPathway.GRU,),
            mode="thermal_throttle",
        )
    return None
