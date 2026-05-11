from __future__ import annotations

import pytest

from src.complexity.soh_monitor import SoHMonitor, SystemHealthSnapshot, TegrastatsProvider


class StaticProvider:
    def __init__(self, snapshot: SystemHealthSnapshot) -> None:
        self.snapshot = snapshot

    def sample(self) -> SystemHealthSnapshot:
        return self.snapshot


def test_tegrastats_parser_extracts_ram_gpu_temp_and_utilization() -> None:
    raw = (
        "RAM 2199/7620MB (lfb 1260x4MB) SWAP 0/3810MB CPU [2%@729,off,off,0%@729] "
        "EMC_FREQ 0%@2133 GR3D_FREQ 42%@306 GPU@54.5C"
    )

    snapshot = TegrastatsProvider.parse(raw)

    assert snapshot.ram_used_mb == pytest.approx(2199.0)
    assert snapshot.ram_total_mb == pytest.approx(7620.0)
    assert snapshot.gpu_utilization_pct == pytest.approx(42.0)
    assert snapshot.gpu_temp_c == pytest.approx(54.5)


def test_soh_budget_is_high_for_healthy_snapshot() -> None:
    snapshot = SystemHealthSnapshot(
        ram_used_mb=2000.0,
        ram_total_mb=8000.0,
        gpu_temp_c=45.0,
        gpu_utilization_pct=20.0,
    )

    assert SoHMonitor.compute_budget(snapshot) == pytest.approx(1.0)


def test_soh_budget_decreases_under_pressure() -> None:
    healthy = SystemHealthSnapshot(
        ram_used_mb=2000.0,
        ram_total_mb=8000.0,
        gpu_temp_c=45.0,
        gpu_utilization_pct=20.0,
    )
    hot_gpu = SystemHealthSnapshot(
        ram_used_mb=2000.0,
        ram_total_mb=8000.0,
        gpu_temp_c=82.0,
        gpu_utilization_pct=20.0,
    )
    high_ram = SystemHealthSnapshot(
        ram_used_mb=7600.0,
        ram_total_mb=8000.0,
        gpu_temp_c=45.0,
        gpu_utilization_pct=20.0,
    )
    saturated_gpu = SystemHealthSnapshot(
        ram_used_mb=2000.0,
        ram_total_mb=8000.0,
        gpu_temp_c=45.0,
        gpu_utilization_pct=98.0,
    )

    assert SoHMonitor.compute_budget(hot_gpu) < SoHMonitor.compute_budget(healthy)
    assert SoHMonitor.compute_budget(high_ram) == pytest.approx(0.0)
    assert SoHMonitor.compute_budget(saturated_gpu) < 0.1


def test_soh_monitor_samples_provider() -> None:
    snapshot = SystemHealthSnapshot(
        ram_used_mb=5600.0,
        ram_total_mb=8000.0,
        gpu_temp_c=65.0,
        gpu_utilization_pct=75.0,
    )

    reading = SoHMonitor(provider=StaticProvider(snapshot)).sample()

    assert reading.snapshot == snapshot
    assert 0.0 <= reading.soh_budget <= 1.0


def test_soh_monitor_rejects_invalid_ram_total() -> None:
    with pytest.raises(ValueError):
        SoHMonitor.compute_budget(SystemHealthSnapshot(ram_used_mb=1.0, ram_total_mb=0.0))


def test_tegrastats_parser_requires_ram_usage() -> None:
    with pytest.raises(ValueError):
        TegrastatsProvider.parse("GR3D_FREQ 42%@306 GPU@54.5C")
