# PLAN: Adaptive Context-Aware Perception System

> Master plan — Robot Mecanum + Jetson Orin Nano 8GB + Raspberry Pi 4

---

## Goal

Xây dựng hệ thống AI nhận thức bối cảnh thích ứng (adaptive) trên robot di động, tự động điều chỉnh độ phức tạp AI theo tình huống thực tế, chạy realtime trên Jetson Orin Nano 8GB trong Docker container.

## Project Type

**EMBEDDED EDGE AI** — Autonomous robot perception system

## Success Criteria

| #   | Criteria             | Metric                                      | Threshold Source |
| --- | -------------------- | ------------------------------------------- | ---------------- |
| 1   | Realtime inference   | ≥20 FPS end-to-end trên Jetson              | `benchmarking.md` §4 |
| 2   | Adaptive routing     | Complexity estimator chọn đúng pathway ≥85% | `benchmarking.md` §4 |
| 3   | Latency proportional | LOW scene <15ms, HIGH scene <40ms           | `benchmarking.md` §4 |
| 4   | Fault tolerance      | System tự recover trong <5s                 | `safety-state-machine.md` §8 |
| 5   | Docker deployment    | 1-command deploy, auto-rebuild on push      | — |
| 6   | GPU RAM              | Peak usage <6.5GB (của 8GB shared)          | `benchmarking.md` §4 |
| 7   | Anomaly detection    | Recall ≥80% trên test set                   | `benchmarking.md` §4 |

---

## Specification Documents

> **All specs MUST be finalized before implementation (Phase -1).**

| Spec | Path | Purpose |
|------|------|---------|
| Communication Protocol | `docs/specs/communication-protocol.md` | Packet format, ports, heartbeat, command schema, retry/timeout |
| Data Schema | `docs/specs/data-schema.md` | Sensor formats, HDF5 record, annotation schema, model I/O contracts |
| Benchmarking | `docs/specs/benchmarking.md` | CI regression vs Jetson hardware, thresholds, measurement methodology |
| Safety State Machine | `docs/specs/safety-state-machine.md` | NORMAL/DEGRADED/ESTOP/RECOVERY states, transition rules, motor stop ownership |

---

## Tech Stack

| Component     | Technology                        | Rationale                 |
| ------------- | --------------------------------- | ------------------------- |
| AI Inference  | TensorRT + ONNX Runtime           | Best perf trên Jetson     |
| GPU Compute   | CUDA 12.x                         | Direct GPU access         |
| Detection     | YOLOv8-s                          | Proven, TRT export mature |
| Tracking      | BoT-SORT                          | Best accuracy/speed ratio |
| RL Policy     | PPO (Stable-Baselines3)           | Stable, discrete action   |
| Language      | Python 3.10 + C++ (perf-critical) | Team familiarity + speed  |
| Container     | Docker + NVIDIA Container Toolkit | Reproducibility           |
| CI/CD         | GitHub Actions                    | Automation                |
| Monitoring    | Custom metrics + logging          | Lightweight for edge      |
| Communication | TCP/UDP sockets                   | Lowest latency            |

---

## File Structure

```
context-aware/
├── config/
│   ├── dev.yaml
│   ├── test.yaml
│   ├── prod.yaml
│   └── schemas/
│       └── annotation_schema.json    # Phase 0.5
├── docker/
│   ├── Dockerfile.dev
│   ├── Dockerfile.test
│   ├── Dockerfile.prod
│   └── docker-compose.yml
├── env/
│   ├── dev/
│   ├── test/
│   └── prod/
├── src/
│   ├── perception/          # Detection, tracking, sensor fusion
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── lidar_proc.py
│   │   ├── depth_proc.py
│   │   ├── imu_fusion.py
│   │   └── sensor_fusion.py
│   ├── complexity/          # Complexity estimator
│   │   ├── estimator.py
│   │   └── soh_monitor.py
│   ├── router/              # Adaptive router + RL policy
│   │   ├── adaptive_router.py
│   │   └── rl_policy.py
│   ├── reasoning/           # GRU, Attention, GNN pathways
│   │   ├── gru_pathway.py
│   │   ├── attention_pathway.py
│   │   ├── gnn_pathway.py
│   │   └── fusion.py
│   ├── decision/            # Intent, anomaly, navigation
│   │   ├── intent_predictor.py
│   │   ├── anomaly_detector.py
│   │   └── nav_commander.py
│   ├── comm/                # RPi ↔ Jetson communication
│   │   ├── protocol.py          # Packet envelope, message types
│   │   ├── lidar_receiver.py
│   │   ├── command_sender.py
│   │   └── health_monitor.py
│   ├── safety/              # Fault tolerance, E-stop
│   │   ├── state_machine.py     # Safety FSM
│   │   ├── watchdog.py
│   │   ├── graceful_degrade.py
│   │   ├── estop.py
│   │   └── recovery.py         # Self-test suite
│   ├── monitoring/          # Metrics, logging
│   │   ├── metrics.py
│   │   └── logger.py
│   ├── utils/               # Shared utilities
│   │   ├── enums.py             # Activity, SceneContext, IntentDirection
│   │   ├── hdf5_recorder.py     # Phase 0.5
│   │   └── hdf5_reader.py       # Phase 0.5
│   └── main.py              # Entry point
├── models/                  # ONNX + TensorRT engines
│   ├── onnx/
│   └── engines/             # .engine built on Jetson, cached in Docker volume
├── scripts/
│   ├── build_engines.sh     # trtexec wrapper
│   ├── benchmark.py         # --device ci | --device jetson
│   ├── update_ci_baselines.py
│   ├── export_onnx.py
│   ├── download_fixtures.py
│   └── generate_synthetic_fixtures.py
├── pipelines/               # Training pipelines (desktop)
│   ├── train_detector.py
│   ├── train_router_rl.py
│   ├── train_gnn.py
│   └── train_attention.py
├── data/                    # Dataset (gitignored)
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── benchmark/
│   │   ├── baselines/       # CI baseline JSON files
│   │   └── test_baselines.py
│   └── fixtures/            # Phase 0.5 bootstrap data
│       ├── coco_person_100/
│       ├── mot17_subset/
│       ├── anomaly_synthetic/
│       ├── rl_scenarios/
│       ├── annotations/
│       └── sample_recording.h5
├── .github/workflows/
│   ├── ci.yml
│   ├── build.yml
│   ├── deploy.yml
│   ├── benchmark-ci.yml     # CPU regression checks
│   ├── benchmark-jetson.yml # Manual trigger, Jetson hardware
│   └── security.yml
├── docs/
│   ├── architecture/
│   │   └── system-design.md
│   ├── plan/
│   │   ├── PLAN-adaptive-context-aware.md
│   │   ├── PLAN-phase-0-infra.md
│   │   ├── PLAN-phase-05-data-foundation.md
│   │   ├── PLAN-phase-1-perception.md
│   │   ├── PLAN-phase-2-adaptive-core.md
│   │   ├── PLAN-phase-3-behavior.md
│   │   ├── PLAN-phase-4-rl-policy.md
│   │   ├── PLAN-phase-5-safety.md
│   │   ├── PLAN-phase-6-optimization.md
│   │   └── PLAN-phase-7-data-mlops.md
│   ├── specs/
│   │   ├── communication-protocol.md
│   │   ├── data-schema.md
│   │   ├── benchmarking.md
│   │   └── safety-state-machine.md
│   └── guide/
├── pyproject.toml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Phase Overview

| Phase | Name                | Focus                                    | Duration (est.) | Plan File                                    |
| ----- | ------------------- | ---------------------------------------- | --------------- | -------------------------------------------- |
| **-1** | Specifications     | 4 spec documents (no code)               | ~1 week         | *(this master plan)*                          |
| **0** | Infrastructure      | Docker, CI/CD, project scaffold          | 1-2 weeks       | `PLAN-phase-0-infra.md`                       |
| **0.5** | Data Foundation   | Schema, HDF5, fixtures, model contracts  | ~1 week         | `PLAN-phase-05-data-foundation.md`            |
| **1** | Perception          | Detection + Tracking + Sensor fusion     | 2-3 weeks       | `PLAN-phase-1-perception.md`                  |
| **2** | Adaptive Core       | Complexity Estimator + Router + Pathways | 3-4 weeks       | `PLAN-phase-2-adaptive-core.md`               |
| **3** | Behavior & Decision | Intent + Anomaly + Navigation            | 2-3 weeks       | `PLAN-phase-3-behavior.md`                    |
| **4** | RL Policy           | Train PPO router + online adaptation     | 2-3 weeks       | `PLAN-phase-4-rl-policy.md`                   |
| **5** | Safety & Monitoring | Fault tolerance + FSM + watchdog         | 1-2 weeks       | `PLAN-phase-5-safety.md`                      |
| **6** | Optimization        | TensorRT INT8, memory, latency tuning    | 2-3 weeks       | `PLAN-phase-6-optimization.md`                |
| **7** | Data & MLOps        | Collection, annotation, registry, tracking | Ongoing       | `PLAN-phase-7-data-mlops.md`                  |

**Total estimated:** 16-23 weeks (~4-6 months)

---

## Phase Dependencies

```
Phase -1 (Specifications)
    │
    └──► Phase 0 (Infra)
             │
             └──► Phase 0.5 (Data Foundation)
                      │
                      ├──► Phase 1 (Perception)
                      │        │
                      │        ├──► Phase 2 (Adaptive Core)
                      │        │        │
                      │        │        ├──► Phase 3 (Behavior)
                      │        │        │        │
                      │        │        │        └──► Phase 4 (RL Policy)
                      │        │        │
                      │        │        └──► Phase 5 (Safety) [parallel w/ Phase 3]
                      │        │
                      │        └──► Phase 7 (Data Collection & MLOps) [starts early, ongoing]
                      │
                      └──► Phase 6 (Optimization) [after Phase 4 complete]
```

---

## Phase X: Verification (Final)

- [ ] All unit tests pass: `pytest tests/unit/`
- [ ] Integration tests pass: `pytest tests/integration/`
- [ ] Jetson benchmark passes: `python scripts/benchmark.py --device jetson --frames 1000`
  - FPS ≥20, GPU RAM <6.5GB, p95 <50ms
- [ ] CI benchmark passes: `python scripts/benchmark.py --device ci --compare-baseline`
  - Latency <15% regression, RSS <10% regression
- [ ] Docker prod image builds: `docker build -f docker/Dockerfile.prod -t ctx-aware:prod .`
- [ ] Engine cache works: restart container → `.engine` loaded from volume (no rebuild)
- [ ] ONNX fallback works: delete `.engine` + fail `trtexec` → ONNX Runtime inference succeeds
- [ ] Safety FSM: all transitions match `safety-state-machine.md` §4
- [ ] E-Stop: heartbeat loss → motors stop <2s
- [ ] Anomaly detection recall ≥80% on bootstrap test set
- [ ] Security scan (Trivy) passes
- [ ] All CI/CD workflows green

---

## Risks

| Risk                                | Impact | Mitigation                                         |
| ----------------------------------- | ------ | -------------------------------------------------- |
| 8GB RAM not enough for all pathways | HIGH   | Lazy loading: only load active pathway weights     |
| TensorRT build fails on Jetson      | MED    | Cache .engine in Docker volume, fallback to ONNX Runtime |
| RL policy doesn't converge          | MED    | Start with rule-based router, add RL incrementally |
| LiDAR TCP latency too high          | MED    | Optimize packet size, match `communication-protocol.md` spec |
| Camera + LiDAR sync drift           | MED    | Timestamp-based sync per `data-schema.md` §2       |
| RPi failure leaves motors running   | HIGH   | Tier 2 hardware watchdog (see `safety-state-machine.md` §2) |

---

## Notes

### Docker / TensorRT Strategy

- Docker image ships **ONNX models only** (architecture-independent)
- `entrypoint.sh` runs `trtexec` on first boot to build `.engine` files
- Built `.engine` files cached in **Docker volume** (`ctx-aware-engines:/app/models/engines`)
- Subsequent starts skip build if valid `.engine` exists and ONNX hash matches
- If TensorRT build fails → **fallback to ONNX Runtime** (degraded performance, logged as WARNING)
- `.engine` files are architecture-specific: MUST be built ON the Jetson

### Safety Ownership

- **Jetson is advisory only** — never directly controls motors
- **RPi owns motor stop** (MVP) — heartbeat timeout → zero velocity → GPIO disable
- **Hardware MCU/relay** (production) — final safety net if RPi also fails
- See `docs/specs/safety-state-machine.md` for complete contract

### Benchmarking Split

- **CI** (`benchmark-ci.yml`): regression detection — CPU latency, RSS, output parity
- **Jetson** (`benchmark-jetson.yml`): absolute performance — FPS, GPU RAM, latency percentiles
- See `docs/specs/benchmarking.md` for methodology and thresholds

### Data Strategy

- Training on desktop GPU, inference on Jetson
- Dataset schema defined in Phase 0.5, collection in Phase 7
- Bootstrap thresholds use public datasets (COCO, MOT17)
- Production thresholds use custom robot data (Phase 7)
