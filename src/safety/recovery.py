from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class RecoveryCheckResult:
    passed: bool
    failed_checks: tuple[str, ...]


def run_recovery_checks(checks: dict[str, Callable[[], bool]]) -> RecoveryCheckResult:
    failed = tuple(name for name, checker in checks.items() if not checker())
    return RecoveryCheckResult(passed=not failed, failed_checks=failed)
