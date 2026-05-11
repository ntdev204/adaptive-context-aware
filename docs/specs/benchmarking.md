# Benchmarking Specification

> Two-tier strategy: CI regression detection + Jetson hardware truth

---

## 1. Strategy Overview

| Tier | Environment | Purpose | Trigger |
|------|-------------|---------|---------|
| **CI** | GitHub Actions (CPU) | Regression detection — flag if metrics deviate from baseline | Every PR |
| **Jetson** | Jetson Orin Nano 8GB | Absolute performance — real FPS, GPU RAM, latency, thermal | Manual / pre-release |

**Principle:** CI catches regressions fast and cheap. Jetson validates real-world performance. Neither replaces the other.

---

## 2. CI Benchmarks (`benchmark-ci.yml`)

### What CI Validates

| Metric | Method | Pass/Fail |
|--------|--------|-----------|
| TensorRT engine output shape parity | Load each model contract, run dummy input, check output shape | Shape matches `data-schema.md` §5 |
| TensorRT engine output value parity | Compare outputs vs stored reference (tolerance 1e-3 in CI, 1e-2 on Jetson FP16) | Max abs diff within tier tolerance |
| CPU relative latency | Run 100 inferences per model on CI CPU, compare vs stored baseline | Fail if >15% slower |
| Peak RSS | Measure peak RSS during full pipeline run (1000 frames synthetic) | Fail if >10% higher than baseline |
| Protocol/schema tests | Run `pytest tests/unit/test_protocol.py tests/unit/test_data_schema.py` | All pass |
| Synthetic pipeline integration | End-to-end pipeline with fixture data, no GPU | Exit 0, no crashes |

### CI Baseline Files

```
tests/benchmark/baselines/
├── latency_baseline.json      # per-model CPU latency reference
├── memory_baseline.json       # peak RSS reference
├── output_reference/          # stored model outputs for parity check
│   ├── yolo11s_ref.npy
│   ├── gru_ref.npy
│   ├── tcn_ref.npy
│   ├── attention_ref.npy
│   ├── gnn_ref.npy
│   ├── estimator_ref.npy
│   └── rl_policy_ref.npy
└── baseline_meta.json         # when/where baselines were captured
```

**`latency_baseline.json` format:**

```json
{
  "captured_on": "ci-runner",
  "captured_at": "2026-05-15T10:00:00Z",
  "ci_runner": "ubuntu-24.04",
  "capture_note": "baselines MUST be captured on CI runner (same hardware class), not Jetson",
  "models": {
    "yolo11s": {"ci_cpu_ms": 45.2, "tolerance_pct": 15},
    "gru_pathway": {"ci_cpu_ms": 0.8, "tolerance_pct": 15},
    "tcn_pathway": {"ci_cpu_ms": 1.2, "tolerance_pct": 15},
    "attention_pathway": {"ci_cpu_ms": 3.1, "tolerance_pct": 15},
    "gnn_pathway": {"ci_cpu_ms": 8.5, "tolerance_pct": 15},
    "complexity_estimator": {"ci_cpu_ms": 0.2, "tolerance_pct": 15},
    "rl_policy": {"ci_cpu_ms": 0.1, "tolerance_pct": 15}
  }
}
```

### CI Benchmark Command

```bash
# Run CI regression benchmark
python scripts/benchmark.py --device ci --compare-baseline

# Exit codes:
#   0 = all within thresholds
#   1 = regression detected (details in stdout)
#   2 = baseline file missing or invalid
```

---

## 3. Jetson Benchmarks (`benchmark-jetson.yml`)

### What Jetson Validates

| Metric | Method | Target | Notes |
|--------|--------|--------|-------|
| End-to-end FPS | 1000 frames, real camera | ≥20 FPS | Mean ± std |
| TensorRT build | `trtexec` all models | Success | Architecture-specific |
| GPU RAM peak | `tegrastats` / `pynvml` during benchmark | <6.5GB | All pathways active |
| Latency p50 | `time.perf_counter_ns()` per frame | <30ms | |
| Latency p95 | Same | <50ms | |
| Latency p99 | Same | <60ms | |
| GPU temperature | `tegrastats` sustained 5min | <80°C | Throttle check |
| Power draw | `jtop` average over 1000 frames | Logged (no threshold) | For reference |

### Measurement Methodology

#### FPS Measurement

```python
# Warmup: 100 frames (discarded)
# Measurement: 1000 frames
# Report: mean, std, min, max FPS
# Clock: time.perf_counter_ns()
# Pipeline: full end-to-end (camera → perception → router → pathway → decision)

warmup_frames = 100
measure_frames = 1000
```

#### Latency Measurement

```python
# Per-module breakdown + end-to-end
# Instrument each module entry/exit with perf_counter_ns()
# Collect 1000 samples
# Report: p50, p95, p99, mean, max
# Optional: nsight systems trace for deep profiling

modules = ["detector", "tracker", "depth_proc", "lidar_proc",
           "imu_fusion", "sensor_fusion", "estimator", "router",
           "gru", "tcn", "attention", "gnn", "fusion", "decision"]
```

#### GPU RAM Measurement

```python
# Method 1: tegrastats parser (parse /sys/ or tegrastats output)
# Method 2: pynvml (if available on Jetson)
# Measure: peak during benchmark, not idle
# Ensure ALL pathways loaded (worst case)
# Report: peak_mb, idle_mb, delta_mb
```

### Jetson Benchmark Command

```bash
# Full Jetson hardware benchmark
python scripts/benchmark.py --device jetson --frames 1000

# Output: JSON report + console summary
# Report saved to: benchmarks/jetson_report_YYYYMMDD_HHMMSS.json
```

---

## 4. Bootstrap Threshold Table

### Detection (YOLOv11-s)

| Metric | Bootstrap Target | Dataset | Upgrade When |
|--------|-----------------|---------|--------------|
| mAP@0.5 (person) | ≥0.60 | COCO person subset (100 imgs) | Custom dataset >500 frames |
| Inference latency | <8ms (Jetson TRT FP16) | Benchmark run | — |

### Tracking (BoT-SORT)

| Metric | Bootstrap Target | Dataset | Upgrade When |
|--------|-----------------|---------|--------------|
| MOTA | ≥0.40 | MOT17-02, MOT17-09 | Custom robot clips >10 sequences |
| ID switches per 100 frames | <5 | Same | Same |
| Tracking latency | <3ms (Jetson) | Benchmark run | — |

### Anomaly Detection

| Metric | Bootstrap Target | Dataset | Upgrade When |
|--------|-----------------|---------|--------------|
| Recall | ≥0.80 | 20 synthetic cases | Robot anomaly clips available |
| Precision | ≥0.50 | Same | Same |
| False positive rate | <0.10 | Same | Same |

### RL Routing Policy

| Metric | Bootstrap Target | Dataset | Upgrade When |
|--------|-----------------|---------|--------------|
| Composite score vs rule-based | >5% improvement | 5 fixed synthetic scenarios | RL environment operational |
| Policy inference latency | <1ms (Jetson) | Benchmark run | — |
| Correct routing (known scenarios) | ≥4/5 scenarios | Same | Same |

### System-Level

| Metric | Target | Tier |
|--------|--------|------|
| End-to-end FPS | ≥20 | Jetson |
| GPU RAM peak | <6.5GB | Jetson |
| Latency p95 | <50ms | Jetson |
| CI CPU latency regression | <15% vs baseline | CI |
| CI peak RSS regression | <10% vs baseline | CI |

---

## 5. CI Baseline Management

### Creating/Updating Baselines

```bash
# Run ON CI runner (or identical hardware class), then export:
# IMPORTANT: CI baselines MUST be captured on the same CPU class as GitHub Actions runner.
# Running on Jetson or a different machine will produce meaningless comparisons.
python scripts/update_ci_baselines.py --source ci-runner --output tests/benchmark/baselines/

# What it does:
# 1. Runs full benchmark on current hardware
# 2. Exports latency_baseline.json, memory_baseline.json
# 3. Generates model output references (*.npy)
# 4. Updates baseline_meta.json with timestamp and source
```

### When to Update Baselines

| Trigger | Action |
|---------|--------|
| New model version pushed | Re-run baseline capture on CI runner; validate `.engine` on Jetson |
| Major dependency update | Re-run baseline capture |
| New release tag | Mandatory baseline refresh |
| CI false positives (>3 in a row) | Investigate, then refresh if legitimate |

### Baseline Staleness

- Baselines older than 30 days trigger a CI **warning** (not failure)
- Baselines older than 90 days trigger a CI **failure** with message: "Stale baseline — run `update_ci_baselines.py`"

---

## 6. Report Format

### Jetson Benchmark Report (`benchmarks/jetson_report_*.json`)

```json
{
  "device": "jetson-orin-nano-8gb",
  "jetpack": "6.0",
  "date": "2026-05-15T10:00:00Z",
  "pipeline": {
    "fps": {"mean": 22.1, "std": 1.3, "min": 18.5, "max": 25.0},
    "latency_ms": {"p50": 28.0, "p95": 45.0, "p99": 52.0, "max": 60.0},
    "gpu_ram_mb": {"peak": 5800, "idle": 2100},
    "gpu_temp_c": {"mean": 62, "max": 71},
    "power_w": {"mean": 8.5}
  },
  "per_module": {
    "detector": {"latency_ms": {"p50": 7.5, "p95": 8.2}},
    "tracker": {"latency_ms": {"p50": 2.8, "p95": 3.1}}
  },
  "models": {
    "yolo11s": {"format": "TRT_FP16", "size_mb": 14},
    "gru": {"format": "TRT_FP16", "size_mb": 0.15},
    "tcn": {"format": "TRT_FP16", "size_mb": 0.23}
  },
  "pass": true,
  "failures": []
}
```

---

## 7. Verification

| Check | Command | Pass/Fail |
|-------|---------|-----------|
| CI benchmark runs | `python scripts/benchmark.py --device ci --compare-baseline` | Exit 0 |
| Jetson benchmark runs | `python scripts/benchmark.py --device jetson --frames 1000` | All metrics pass §4 |
| Baseline JSON valid | `pytest tests/benchmark/test_baselines.py` | Schema valid, not stale |
| Baseline update works | `python scripts/update_ci_baselines.py --source mock` | Files created, schema valid |
