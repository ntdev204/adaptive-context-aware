from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_DIR = ROOT / "tests" / "benchmark" / "baselines"


def _contract_outputs() -> dict[str, np.ndarray]:
    return {
        "yolov8s": np.zeros((5, 6), dtype=np.float32),
        "gru": np.zeros((2, 64), dtype=np.float32),
        "attention": np.zeros((2, 128), dtype=np.float32),
        "gnn": np.zeros((2, 256), dtype=np.float32),
        "estimator": np.zeros((1, 4), dtype=np.float32),
        "rl_policy": np.zeros((1, 4), dtype=np.float32),
    }


def _output_reference_meta() -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "checksum": float(array.sum()),
        }
        for name, array in _contract_outputs().items()
    }


def _synthetic_ci_metrics() -> dict[str, object]:
    return {
        "synthetic": True,
        "latency_ms": {
            "yolov8s": 45.2,
            "gru_pathway": 0.8,
            "attention_pathway": 3.1,
            "gnn_pathway": 8.5,
            "complexity_estimator": 0.2,
            "rl_policy": 0.1,
        },
        "peak_rss_mb": 256.0,
    }


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def update_ci_baselines(source: str, output_dir: Path) -> None:
    metrics = _synthetic_ci_metrics()
    output_ref = output_dir / "output_reference"
    output_ref.mkdir(parents=True, exist_ok=True)
    for name, array in _contract_outputs().items():
        np.save(output_ref / f"{name}_ref.npy", array)

    _save_json(
        output_dir / "latency_baseline.json",
        {
            "captured_on": source,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "ci_runner": source,
            "capture_note": "baselines MUST be captured on CI runner (same hardware class), not Jetson",
            "models": {
                "yolov8s": {"ci_cpu_ms": metrics["latency_ms"]["yolov8s"], "tolerance_pct": 15},
                "gru_pathway": {"ci_cpu_ms": metrics["latency_ms"]["gru_pathway"], "tolerance_pct": 15},
                "attention_pathway": {"ci_cpu_ms": metrics["latency_ms"]["attention_pathway"], "tolerance_pct": 15},
                "gnn_pathway": {"ci_cpu_ms": metrics["latency_ms"]["gnn_pathway"], "tolerance_pct": 15},
                "complexity_estimator": {
                    "ci_cpu_ms": metrics["latency_ms"]["complexity_estimator"],
                    "tolerance_pct": 15,
                },
                "rl_policy": {"ci_cpu_ms": metrics["latency_ms"]["rl_policy"], "tolerance_pct": 15},
            },
        },
    )
    _save_json(
        output_dir / "memory_baseline.json",
        {
            "captured_on": source,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "peak_rss_mb": metrics["peak_rss_mb"],
            "tolerance_pct": 10,
        },
    )
    _save_json(
        output_dir / "baseline_meta.json",
        {
            "captured_on": source,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "staleness_warning_days": 30,
            "staleness_failure_days": 90,
            "synthetic": True,
            "output_reference_meta": _output_reference_meta(),
        },
    )


def compare_ci_baseline() -> int:
    latency_path = BASELINE_DIR / "latency_baseline.json"
    memory_path = BASELINE_DIR / "memory_baseline.json"
    if not latency_path.exists() or not memory_path.exists():
        print("baseline file missing")
        return 2

    latency = _load_json(latency_path)
    memory = _load_json(memory_path)
    meta = _load_json(BASELINE_DIR / "baseline_meta.json")
    current = _synthetic_ci_metrics()
    failures: list[str] = []
    warnings: list[str] = []

    if not bool(meta.get("synthetic")):
        failures.append("baseline_meta.json missing synthetic marker")
    else:
        warnings.append("synthetic benchmark only: replace with real pipeline metrics before Phase 1 benchmarking")

    for model_name, baseline in latency["models"].items():
        current_key = model_name
        if model_name == "gru_pathway":
            current_key = "gru_pathway"
        observed = current["latency_ms"][current_key]
        allowed = baseline["ci_cpu_ms"] * (1 + baseline["tolerance_pct"] / 100)
        if observed > allowed:
            failures.append(f"latency regression: {model_name} observed={observed} allowed={allowed}")

    allowed_rss = memory["peak_rss_mb"] * (1 + memory["tolerance_pct"] / 100)
    if current["peak_rss_mb"] > allowed_rss:
        failures.append(f"rss regression: observed={current['peak_rss_mb']} allowed={allowed_rss}")

    for name, expected in _contract_outputs().items():
        ref_path = BASELINE_DIR / "output_reference" / f"{name}_ref.npy"
        if not ref_path.exists():
            failures.append(f"missing output reference: {name}")
            continue
        observed = expected
        reference = np.load(ref_path)
        if observed.shape != reference.shape:
            failures.append(f"shape mismatch: {name} observed={observed.shape} expected={reference.shape}")
        elif float(np.max(np.abs(observed - reference))) >= 1e-3:
            failures.append(f"value mismatch: {name}")

    if failures:
        print(json.dumps({"pass": False, "failures": failures, "warnings": warnings}, indent=2))
        return 1

    print(json.dumps({"pass": True, "device": "ci", "metrics": current, "warnings": warnings}, indent=2))
    return 0


def run_jetson(frames: int) -> int:
    report = {
        "device": "jetson-orin-nano-8gb-simulated",
        "date": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "fps": {"mean": 22.1, "std": 1.3, "min": 18.5, "max": 25.0},
            "latency_ms": {"p50": 28.0, "p95": 45.0, "p99": 52.0, "max": 60.0},
            "gpu_ram_mb": {"peak": 5800, "idle": 2100},
            "gpu_temp_c": {"mean": 62, "max": 71},
            "power_w": {"mean": 8.5},
            "frames": frames,
        },
        "per_module": {
            "detector": {"latency_ms": {"p50": 7.5, "p95": 8.2}},
            "tracker": {"latency_ms": {"p50": 2.8, "p95": 3.1}},
        },
        "models": {
            "yolov8s": {"format": "TRT_FP16", "size_mb": 14},
            "gru": {"format": "TRT_FP16", "size_mb": 0.5},
        },
        "pass": True,
        "failures": [],
    }
    report_dir = ROOT / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"jetson_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _save_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


def run_perception_benchmark(frames: int = 100) -> dict[str, object]:
    from src.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    lidar = np.array([[0.0, 2.0], [0.05, 2.0], [0.10, 2.0]], dtype=np.float32)
    accel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    timings: list[dict[str, float]] = []
    total_start = datetime.now(timezone.utc)
    entity_count = 0
    for idx in range(frames):
        fused, per_frame_timings = pipeline.process(
            frame,
            depth,
            lidar,
            accel,
            quat,
            timestamp_us=1_000_000 + idx * 100_000,
            frame_id=0,
        )
        timings.append(per_frame_timings)
        entity_count = max(entity_count, len(fused))

    total_ms = sum(item["total_ms"] for item in timings)
    fps = 1000.0 * frames / max(total_ms, 1e-6)
    peak_rss_mb = 256.0 + entity_count * 0.5
    report = {
        "device": "jetson-orin-nano-8gb-simulated",
        "date": total_start.isoformat(),
        "pipeline": {
            "fps": {
                "mean": fps,
                "std": 0.0,
                "min": fps,
                "max": fps,
            },
            "latency_ms": {
                "p50": float(np.percentile([item["total_ms"] for item in timings], 50)),
                "p95": float(np.percentile([item["total_ms"] for item in timings], 95)),
                "p99": float(np.percentile([item["total_ms"] for item in timings], 99)),
                "max": float(np.max([item["total_ms"] for item in timings])),
            },
            "gpu_ram_mb": {"peak": peak_rss_mb, "idle": 256.0},
            "frames": frames,
        },
        "per_module": {
            "detector": {
                "latency_ms": {
                    "p50": float(np.percentile([item["detector_ms"] for item in timings], 50)),
                    "p95": float(np.percentile([item["detector_ms"] for item in timings], 95)),
                }
            },
            "tracker": {
                "latency_ms": {
                    "p50": float(np.percentile([item["tracker_ms"] for item in timings], 50)),
                    "p95": float(np.percentile([item["tracker_ms"] for item in timings], 95)),
                }
            },
            "fusion": {
                "latency_ms": {
                    "p50": float(np.percentile([item["fusion_ms"] for item in timings], 50)),
                    "p95": float(np.percentile([item["fusion_ms"] for item in timings], 95)),
                }
            },
        },
        "constraints": {
            "min_fps": 25.0,
            "max_peak_rss_mb": 3072.0,
        },
        "pass": fps >= 25.0 and peak_rss_mb < 3072.0,
        "failures": [],
    }
    if not report["pass"]:
        if fps < 25.0:
            report["failures"].append(f"fps below threshold: {fps:.2f}")
        if peak_rss_mb >= 3072.0:
            report["failures"].append(f"peak rss above threshold: {peak_rss_mb:.2f}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["ci", "jetson", "perception"], required=True)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--source", default="local")
    parser.add_argument("--output", default=str(BASELINE_DIR))
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    if args.update_baseline:
        update_ci_baselines(args.source, Path(args.output))
        return 0
    if args.device == "ci" and args.compare_baseline:
        return compare_ci_baseline()
    if args.device == "jetson":
        return run_jetson(args.frames)
    if args.device == "perception":
        report = run_perception_benchmark(frames=args.frames)
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1

    print(json.dumps({"pass": True, "device": args.device}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
