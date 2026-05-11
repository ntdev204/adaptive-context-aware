from __future__ import annotations

from dataclasses import dataclass

from src.utils.enums import EStopReason, EStopSource, SafetyState


@dataclass(frozen=True, slots=True)
class EStopCommand:
    velocity_xyz: tuple[float, float, float]
    motors_enabled: bool
    reason: EStopReason
    source: EStopSource
    state: SafetyState = SafetyState.ESTOP


def build_estop_command(reason: EStopReason, source: EStopSource) -> EStopCommand:
    return EStopCommand(
        velocity_xyz=(0.0, 0.0, 0.0),
        motors_enabled=False,
        reason=reason,
        source=source,
    )
