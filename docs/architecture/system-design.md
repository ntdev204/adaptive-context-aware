# System Design: Adaptive Context-Aware Perception

> Robot di động Mecanum 4 bánh — Jetson Orin Nano 8GB + Raspberry Pi 4

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ROBOT HARDWARE                        │
│                                                         │
│  ┌──────────────┐  TCP/UDP   ┌────────────────────────┐ │
│  │ Raspberry Pi 4│◄────────►│  Jetson Orin Nano 8GB  │ │
│  │              │           │                        │ │
│  │ • LiDAR drv  │           │ • RGB-D Camera         │ │
│  │ • Motor ctrl │           │ • IMU                  │ │
│  │ • Mecanum    │           │ • AI Pipeline (Docker) │ │
│  │ • E-Stop     │           │ • TensorRT / ONNX      │ │
│  └──────────────┘           └────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Architecture: Adaptive AI Pipeline

```
Sensor Input (RGB-D + LiDAR + IMU)
        │
        ▼
┌──────────────────────┐
│   Perception Layer   │  ← Always-on, lightweight
│  (Det + Track + 3D)  │
└──────────┬───────────┘
           │ Entity features + scene embedding
           ▼
┌──────────────────────┐
│ Complexity Estimator │  ← Tiny MLP (~50KB)
│  crowd_density       │
│  motion_entropy      │
│  anomaly_score       │
│  soh_budget          │
└──────────┬───────────┘
           │ complexity_level ∈ {LOW, MED, HIGH, CRITICAL}
           ▼
┌──────────────────────┐
│   Adaptive Router    │  ← RL/DRL Policy (PPO)
│   (learned policy)   │
└───┬──────┬───────┬───┘
    │      │       │
    ▼      ▼       ▼
┌──────┐┌────────┐┌─────┐
│ GRU  ││ Attn   ││ GNN │  ← 3 reasoning pathways
│~1ms  ││ ~5ms   ││~10ms│
│0.5MB ││  2MB   ││ 5MB │
└──┬───┘└───┬────┘└──┬──┘
   │        │        │
   ▼        ▼        ▼
┌──────────────────────┐
│    Fusion Layer      │  ← Weighted merge based on router
│  (gated attention)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Decision Layer     │
│  • Intent prediction │
│  • Anomaly alert     │
│  • Nav command       │
│  • Scene context     │
└──────────────────────┘
```

---

## 3. Complexity Levels & Resource Allocation

| Level | Trigger | Pathway | Latency | GPU% | Example |
|-------|---------|---------|---------|------|---------|
| **LOW** | <5 people, low motion | GRU only | <15ms | 20% | Empty hallway |
| **MED** | 5-15 people, normal | GRU + Attention | <25ms | 50% | Busy lobby |
| **HIGH** | >15 people, fast motion | Attention + GNN | <40ms | 80% | Crowded gate |
| **CRITICAL** | Anomaly detected | All pathways | <50ms | 100% | Emergency |

### SoH (State of Health) Aware

```python
# Resource budget adjusts based on hardware state
soh_factors = {
    "gpu_temp": normalize(gpu_temp, 40, 85),      # throttle at 85°C
    "gpu_util": normalize(gpu_util, 0, 100),
    "ram_free": normalize(ram_free_mb, 500, 4000),
    "battery":  normalize(battery_pct, 10, 100),   # if battery-powered
    "fps_avg":  normalize(fps_30s_avg, 10, 30),
}
soh_budget = weighted_mean(soh_factors)  # 0.0 = critical, 1.0 = healthy
```

---

## 4. Hardware Communication

```
┌─────────────────┐         ┌────────────────────┐
│  Raspberry Pi 4 │         │  Jetson Orin Nano   │
│                 │         │                    │
│ LiDAR Driver    │──TCP───►│ LiDAR Receiver     │
│ (scan @ 10Hz)   │  9090   │ (point cloud proc) │
│                 │         │                    │
│ Motor Control   │◄──TCP───│ Navigation Cmd     │
│ (mecanum ctrl)  │  9091   │ (vel_x, vel_y, ω)  │
│                 │         │                    │
│ Health Monitor  │──UDP───►│ SoH Collector      │
│ (temp, voltage) │  9092   │ (resource monitor)  │
│                 │         │                    │
│ E-Stop Handler  │◄──TCP───│ Safety Controller   │
│ (kill switch)   │  9093   │ (fault tolerance)   │
└─────────────────┘         └────────────────────┘
```

**Protocol:** TCP for reliable data (LiDAR, commands), UDP for telemetry (SoH).

> **Full specification:** See `docs/specs/communication-protocol.md` for packet format, message types, heartbeat contract, retry/timeout policy, and sequence diagrams.

---

## 5. Perception Layer Detail

| Component | Model | Format | Size | Latency | Input |
|-----------|-------|--------|------|---------|-------|
| Detection | YOLOv8-s | TensorRT FP16 | ~14MB | ~8ms | RGB 640×480 |
| Tracking | BoT-SORT | C++/Python | ~2MB | ~3ms | Detections + depth |
| Depth Proc | Custom | CUDA kernel | <1MB | ~2ms | Depth map 640×480 |
| LiDAR Proc | Custom | NumPy/CUDA | <1MB | ~2ms | 2D scan points |
| IMU Fusion | EKF | C++ | <1MB | <1ms | Accel + Gyro |
| Pose (opt) | MoveNet | TensorRT INT8 | ~5MB | ~4ms | RGB crop |
| **Total** | | | **~23MB** | **~20ms** | |

---

## 6. Adaptive Router (RL/DRL Policy)

### State Space
```python
state = {
    "crowd_density": float,       # 0-1 normalized
    "motion_entropy": float,      # 0-1 (how chaotic)
    "anomaly_probability": float, # 0-1
    "soh_budget": float,          # 0-1 (available resources)
    "scene_embedding": float[32], # compressed scene vector
    "prev_action": int,           # previous routing decision
    "time_since_critical": float, # seconds since last anomaly
}
```

### Action Space
```python
actions = {
    0: "GRU_ONLY",           # fastest, lowest accuracy
    1: "GRU_ATTENTION",      # balanced
    2: "ATTENTION_GNN",      # high accuracy, slower
    3: "ALL_PATHWAYS",       # maximum accuracy
}
```

### Reward Function
```python
reward = (
    α * accuracy_score          # correct behavior prediction
    - β * latency_penalty       # penalize slow inference
    - γ * energy_cost           # penalize GPU usage
    + δ * anomaly_catch_bonus   # bonus for catching anomalies
    - ε * miss_penalty          # heavy penalty for missing critical events
)
# α=1.0, β=0.3, γ=0.1, δ=2.0, ε=5.0 (tunable)
```

### Training Strategy
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Training:** Offline on desktop GPU → export ONNX → TensorRT on Jetson
- **Online adaptation:** Lightweight policy update on Jetson (optional)

---

## 7. Docker Architecture

```
┌─────────────────────────────────────────┐
│         Jetson Orin Nano (Host)         │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │     Docker (NVIDIA Runtime)       │  │
│  │                                   │  │
│  │  ┌─────────────┐ ┌────────────┐  │  │
│  │  │ ai-pipeline │ │  monitor   │  │  │
│  │  │             │ │            │  │  │
│  │  │ • TensorRT  │ │ • Metrics  │  │  │
│  │  │ • CUDA 12   │ │ • Logging  │  │  │
│  │  │ • Pipeline  │ │ • Alerts   │  │  │
│  │  └─────────────┘ └────────────┘  │  │
│  │                                   │  │
│  │  ┌─────────────┐ ┌────────────┐  │  │
│  │  │  comm-layer │ │  watchdog  │  │  │
│  │  │             │ │            │  │  │
│  │  │ • TCP/UDP   │ │ • Health   │  │  │
│  │  │ • LiDAR rx  │ │ • Restart  │  │  │
│  │  │ • Cmd tx    │ │ • E-Stop   │  │  │
│  │  └─────────────┘ └────────────┘  │  │
│  │                                   │  │
│  │  Volume: ctx-aware-engines         │  │
│  │  (.engine cache, persists across   │  │
│  │   container restarts)              │  │
│  └───────────────────────────────────┘  │
│                                         │
│  systemd: docker-compose auto-start     │
└─────────────────────────────────────────┘
```

---

## 8. Environments

| Env | Purpose | Inference Engine | Logging | Safety |
|-----|---------|-----------------|---------|--------|
| **dev** | Local development, hot-reload | ONNX Runtime (CPU/GPU) | DEBUG | Disabled |
| **test** | CI/CD, benchmark, integration | ONNX Runtime (CI) / TensorRT FP16 (Jetson) | INFO | Simulated |
| **prod** | Robot deployment | TensorRT INT8 (fallback: ONNX Runtime) | WARNING | Full E-Stop + FSM |

---

## 9. CI/CD Pipelines

| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci.yml` | PR → main | Lint (ruff) + Unit tests + Type check |
| `build.yml` | Push main | Build Docker images (multi-arch) |
| `deploy.yml` | Tag v*.*.* | Push to registry → Jetson pulls |
| `benchmark-ci.yml` | PR → main | CPU regression: latency, RSS, output parity (see `benchmarking.md` §2) |
| `benchmark-jetson.yml` | Manual / pre-release | Hardware: FPS, GPU RAM, TRT build, latency p50/p95/p99 (see `benchmarking.md` §3) |
| `security.yml` | Weekly + PR | Container scan (Trivy) + dependency audit |

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inter-device comm | Raw TCP/UDP | Lowest latency, no ROS overhead |
| AI framework | TensorRT + ONNX | Best Jetson performance |
| Container | Docker + NVIDIA runtime | Reproducibility + GPU access |
| RL algorithm | PPO | Stable, works with discrete actions |
| Tracking | BoT-SORT | Best accuracy/speed on Jetson |
| Detection | YOLOv8-s | Proven, TensorRT export mature |
| Monitoring | Prometheus + custom | Lightweight, edge-compatible |

---

## 11. Fault Tolerance

> **Safety principle:** Jetson is advisory only. RPi or hardware layer owns final motor stop.
> **Full specification:** See `docs/specs/safety-state-machine.md`

```
┌─────────────────────────────────────┐
│          Fault Tolerance            │
│                                     │
│  Level 1: Watchdog (systemd)        │
│  ├── Container crash → auto restart │
│  └── Health check every 5s          │
│                                     │
│  Level 2: Graceful Degradation      │
│  ├── GPU OOM → fallback to GRU only │
│  ├── Camera fail → LiDAR-only mode  │
│  └── High temp → reduce complexity  │
│                                     │
│  Level 3: Safety State Machine      │
│  ├── States: NORMAL → DEGRADED      │
│  │           → ESTOP → RECOVERY     │
│  ├── Heartbeat loss >2s → ESTOP     │
│  ├── Motor stop owner: RPi (MVP)    │
│  ├── Hardware watchdog (Production) │
│  └── No auto-recovery from ESTOP    │
└─────────────────────────────────────┘
```

---

## 12. Data Pipeline (Data-Centric AI)

```
Robot (Jetson)                    Dev Machine
┌──────────┐                     ┌──────────────┐
│ Capture  │──rsync/scp────────►│ Data Lake     │
│ • RGB-D  │                     │ • Raw frames  │
│ • LiDAR  │                     │ • Annotations │
│ • IMU    │                     │ • Metadata    │
│ • Labels │                     └──────┬───────┘
└──────────┘                            │
                                        ▼
                                 ┌──────────────┐
                                 │ Training      │
                                 │ • Desktop GPU │
                                 │ • Export ONNX │
                                 └──────┬───────┘
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │ Optimize      │
                                 │ • TensorRT    │
                                 │ • INT8 calib  │
                                 │ • .engine     │
                                 └──────┬───────┘
                                        │
                                  push to registry
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │ Deploy        │
                                 │ • Jetson pull │
                                 │ • Rebuild eng │
                                 └──────────────┘
```

> **Note:** `.engine` files MUST be built ON the Jetson (architecture-specific).
> Docker image ships ONNX models only. `entrypoint.sh` runs `trtexec` on first boot.
> Built engines cached in Docker volume (`ctx-aware-engines`). If TRT build fails → fallback to ONNX Runtime.

---

## 13. Specification Documents

| Spec | Path | Key Contents |
|------|------|--------------|
| Communication Protocol | `docs/specs/communication-protocol.md` | Packet format (21B header + CRC-16), 7 message types, heartbeat 500ms/2s timeout, retry policy |
| Data Schema | `docs/specs/data-schema.md` | Sensor formats, HDF5 record structure, annotation JSON Schema, model I/O contracts (9 models) |
| Benchmarking | `docs/specs/benchmarking.md` | CI regression (CPU latency ±15%, RSS ±10%) vs Jetson hardware (FPS ≥20, GPU <6.5GB, p95 <50ms) |
| Safety State Machine | `docs/specs/safety-state-machine.md` | 4 states, 13 transitions, two-tier ownership (RPi MVP / hardware MCU production) |
