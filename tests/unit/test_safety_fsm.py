from __future__ import annotations

import pytest

from src.safety.state_machine import InvalidTransitionError, SafetyStateMachine
from src.utils.enums import SafetyState, StatusChangeReason


def test_valid_transitions_succeed() -> None:
    fsm = SafetyStateMachine()
    assert (
        fsm.request_transition(SafetyState.DEGRADED, StatusChangeReason.GPU_OVERHEAT).new_state == SafetyState.DEGRADED
    )
    assert fsm.request_transition(SafetyState.ESTOP, StatusChangeReason.CAMERA_FAIL).new_state == SafetyState.ESTOP
    assert (
        fsm.request_transition(SafetyState.RECOVERY, StatusChangeReason.CONDITION_RESOLVED).new_state
        == SafetyState.RECOVERY
    )
    assert fsm.request_transition(SafetyState.NORMAL, StatusChangeReason.SELF_TEST_PASS).new_state == SafetyState.NORMAL


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (SafetyState.ESTOP, SafetyState.NORMAL),
        (SafetyState.ESTOP, SafetyState.DEGRADED),
        (SafetyState.RECOVERY, SafetyState.DEGRADED),
        (SafetyState.NORMAL, SafetyState.RECOVERY),
    ],
)
def test_invalid_transitions_raise(start: SafetyState, target: SafetyState) -> None:
    fsm = SafetyStateMachine()
    fsm.state = start
    with pytest.raises(InvalidTransitionError):
        fsm.request_transition(target)
