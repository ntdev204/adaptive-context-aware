from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_ci_compare_baseline() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--device", "ci", "--compare-baseline"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "synthetic benchmark only" in result.stdout


def test_update_ci_baselines_script(tmp_path) -> None:
    output = tmp_path / "baselines"
    result = subprocess.run(
        [sys.executable, "scripts/update_ci_baselines.py", "--source", "test-runner", "--output", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "latency_baseline.json").exists()
    assert (output / "memory_baseline.json").exists()


def test_perception_benchmark_report_contains_thresholds() -> None:
    from scripts.benchmark import run_perception_benchmark

    report = run_perception_benchmark(frames=5)
    assert report["pipeline"]["fps"]["mean"] > 0
    assert report["constraints"]["min_fps"] == 25.0
    assert "detector" in report["per_module"]
    assert "tracker" in report["per_module"]
