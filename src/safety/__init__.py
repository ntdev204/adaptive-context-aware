"""Safety subsystem."""

from .estop import EStopCommand, build_estop_command
from .graceful_degrade import DegradationDecision, decide_degradation
from .logger import LoggerConfig, build_logger
from .metrics import SafetyMetrics
from .recovery import RecoveryCheckResult, run_recovery_checks
from .state_machine import InvalidTransitionError, SafetyStateMachine, TransitionResult
from .watchdog import ProcessWatchdog

__all__ = [
    "DegradationDecision",
    "EStopCommand",
    "InvalidTransitionError",
    "LoggerConfig",
    "ProcessWatchdog",
    "RecoveryCheckResult",
    "SafetyMetrics",
    "SafetyStateMachine",
    "TransitionResult",
    "build_estop_command",
    "build_logger",
    "decide_degradation",
    "run_recovery_checks",
]
