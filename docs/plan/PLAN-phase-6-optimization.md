# Phase 6: Optimization

> TensorRT INT8, memory optimization, latency tuning, model compression

## Goal

Tối ưu toàn bộ pipeline cho production trên Jetson: INT8 quantization, memory pooling, latency profiling.

## Spec Reference

- **Benchmarking:** `docs/specs/benchmarking.md` — methodology, thresholds, CI vs Jetson split
- **Data:** `docs/specs/data-schema.md` — model I/O contracts for shape validation

---

## Tasks

- [ ] **T6.1**: Build tất cả models → TensorRT `.engine` (YOLOv11-s, GRU, TCN, Attention, GNN, RL policy)
  → Verify: TensorRT engines produce expected output shapes and values within tolerance vs PyTorch checkpoints

- [ ] **T6.2**: TensorRT FP16 engine validation — tất cả models
  → Verify: Inference results match PyTorch reference (tolerance <1e-2)

- [ ] **T6.3**: TensorRT INT8 calibration — tạo calibration dataset, quantize
  → Verify: Accuracy drop <2% so với FP16 (per `benchmarking.md` §4)

- [ ] **T6.4**: Memory optimization — CUDA memory pool, model weight sharing, lazy loading
  → Verify: Peak GPU RAM <6.5GB khi ALL pathways active (per `benchmarking.md` §4)

- [ ] **T6.5**: Latency profiling — nsight systems trace per module
  → Verify: Bottleneck identified + optimized

- [ ] **T6.6**: Model pruning/distillation (nếu cần) — giảm size models
  → Verify: Model size giảm ≥30%, accuracy drop <3%

- [ ] **T6.7**: Benchmark script (`scripts/benchmark.py`) — **two modes**
  - `--device ci`: CPU-only regression checks against stored baselines
    - Engine contract output shape/value parity using stored references
    - CPU latency vs baseline (fail if >15% slower)
    - Peak RSS vs baseline (fail if >10% higher)
  - `--device jetson`: full hardware benchmark (1000 frames)
    - FPS (must ≥20), GPU RAM peak (<6.5GB)
    - Latency p50/p95/p99
    - Power consumption, thermal
  - Methodology per `benchmarking.md` §3-4
  → Verify: Both modes run successfully, JSON report generated

- [ ] **T6.8**: `.engine` build automation — `scripts/build_engines.sh`
  - Build all `.engine` files trên Jetson trong 1 command
  - Engine cache in Docker volume (`ctx-aware-engines`)
  - No CPU inference fallback if TensorRT build/load fails
  - Idempotent: skip build if valid `.engine` exists and engine metadata hash matches
  → Verify: Build all engines. Restart → engines loaded from cache. Simulate TRT failure → service enters documented degraded/safe mode

- [ ] **T6.9**: CI baseline management — `scripts/update_ci_baselines.py`
  - Run Jetson benchmark → export results → update `tests/benchmark/baselines/*.json`
  - Baseline JSON schema validation
  - Staleness warning (>30 days) and failure (>90 days)
  → Verify: `pytest tests/benchmark/test_baselines.py` — schema valid, CI regression test passes

## Done When

- [ ] ≥20 FPS end-to-end trên Jetson Orin Nano (per `benchmarking.md` §4)
- [ ] GPU RAM peak <6.5GB (per `benchmarking.md` §4)
- [ ] INT8 accuracy drop <2%
- [ ] CI benchmark passes: `python scripts/benchmark.py --device ci --compare-baseline` exit 0
- [ ] Jetson benchmark passes: `python scripts/benchmark.py --device jetson --frames 1000` all metrics green
- [ ] Engine cache + engine-only degraded/safe failure path working
- [ ] CI baselines up-to-date and schema-valid
