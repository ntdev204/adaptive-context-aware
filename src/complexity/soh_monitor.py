from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class SystemHealthSnapshot:
    ram_used_mb: float
    ram_total_mb: float
    gpu_temp_c: float | None = None
    gpu_utilization_pct: float | None = None


@dataclass(frozen=True, slots=True)
class SoHReading:
    snapshot: SystemHealthSnapshot
    soh_budget: float


class HealthSnapshotProvider(Protocol):
    def sample(self) -> SystemHealthSnapshot:
        """Return one system health snapshot."""


class SoHMonitor:
    def __init__(self, provider: HealthSnapshotProvider | None = None) -> None:
        self.provider = provider or TegrastatsProvider()

    def sample(self) -> SoHReading:
        snapshot = self.provider.sample()
        return SoHReading(snapshot=snapshot, soh_budget=self.compute_budget(snapshot))

    @staticmethod
    def compute_budget(snapshot: SystemHealthSnapshot) -> float:
        if snapshot.ram_total_mb <= 0:
            raise ValueError("ram_total_mb must be positive")

        ram_pressure = snapshot.ram_used_mb / snapshot.ram_total_mb
        scores = [
            _linear_budget(ram_pressure, healthy_at=0.70, critical_at=0.95),
            _linear_budget(snapshot.gpu_temp_c, healthy_at=65.0, critical_at=85.0),
            _linear_budget(snapshot.gpu_utilization_pct, healthy_at=75.0, critical_at=99.0),
        ]
        return float(np.clip(min(scores), 0.0, 1.0))


class TegrastatsProvider:
    RAM_PATTERN = re.compile(r"\bRAM\s+([0-9.]+)/([0-9.]+)MB\b")
    GPU_TEMP_PATTERN = re.compile(r"\bGPU@([0-9.]+)C\b")
    GPU_UTIL_PATTERN = re.compile(r"\bGR3D_FREQ\s+([0-9.]+)%")

    def __init__(
        self,
        command: tuple[str, ...] = ("tegrastats", "--interval", "100", "--count", "1"),
        timeout_s: float = 2.0,
    ) -> None:
        self.command = command
        self.timeout_s = timeout_s

    def sample(self) -> SystemHealthSnapshot:
        completed = subprocess.run(
            self.command,
            capture_output=True,
            check=True,
            text=True,
            timeout=self.timeout_s,
        )
        return self.parse(completed.stdout)

    @classmethod
    def parse(cls, raw_output: str) -> SystemHealthSnapshot:
        line = raw_output.strip().splitlines()[-1] if raw_output.strip() else ""
        ram_match = cls.RAM_PATTERN.search(line)
        if ram_match is None:
            raise ValueError("tegrastats output does not include RAM usage")

        temp_match = cls.GPU_TEMP_PATTERN.search(line)
        util_match = cls.GPU_UTIL_PATTERN.search(line)
        return SystemHealthSnapshot(
            ram_used_mb=float(ram_match.group(1)),
            ram_total_mb=float(ram_match.group(2)),
            gpu_temp_c=float(temp_match.group(1)) if temp_match else None,
            gpu_utilization_pct=float(util_match.group(1)) if util_match else None,
        )


def _linear_budget(value: float | None, *, healthy_at: float, critical_at: float) -> float:
    if value is None:
        return 1.0
    if critical_at <= healthy_at:
        raise ValueError("critical_at must be greater than healthy_at")
    pressure = (value - healthy_at) / (critical_at - healthy_at)
    return float(1.0 - np.clip(pressure, 0.0, 1.0))
