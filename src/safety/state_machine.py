from __future__ import annotations

from dataclasses import dataclass

from src.utils.enums import SafetyState, StatusChangeReason


@dataclass(slots=True)
class TransitionResult:
    old_state: SafetyState
    new_state: SafetyState
    reason: StatusChangeReason | None


class SafetyStateMachine:
    def __init__(self) -> None:
        self.state = SafetyState.NORMAL

    def request_transition(self, target: SafetyState, reason: StatusChangeReason | None = None) -> TransitionResult:
        current = self.state
        if not self._is_valid_transition(current, target):
            raise ValueError(f"invalid transition {current.name}->{target.name}")
        self.state = target
        return TransitionResult(old_state=current, new_state=target, reason=reason)

    @staticmethod
    def _is_valid_transition(current: SafetyState, target: SafetyState) -> bool:
        if current == target:
            return True
        invalid = {
            (SafetyState.ESTOP, SafetyState.NORMAL),
            (SafetyState.ESTOP, SafetyState.DEGRADED),
            (SafetyState.RECOVERY, SafetyState.DEGRADED),
            (SafetyState.NORMAL, SafetyState.RECOVERY),
        }
        return (current, target) not in invalid
