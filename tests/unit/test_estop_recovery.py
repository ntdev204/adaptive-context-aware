from __future__ import annotations

from src.safety.estop import build_estop_command
from src.safety.recovery import run_recovery_checks
from src.utils.enums import EStopReason, EStopSource, SafetyState


def test_estop_command_disables_motors() -> None:
    command = build_estop_command(EStopReason.ANOMALY_CRITICAL, EStopSource.ADAPTIVE_RUNTIME)
    assert command.velocity_xyz == (0.0, 0.0, 0.0)
    assert not command.motors_enabled
    assert command.state == SafetyState.ESTOP


def test_recovery_pass() -> None:
    result = run_recovery_checks({"camera": lambda: True, "depth_camera": lambda: True})
    assert result.passed
    assert result.failed_checks == ()


def test_recovery_fail() -> None:
    result = run_recovery_checks({"camera": lambda: False, "depth_camera": lambda: True})
    assert not result.passed
    assert result.failed_checks == ("camera",)
