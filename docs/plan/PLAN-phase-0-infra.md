# Phase 0: Infrastructure & Project Scaffold

> Docker, CI/CD, communication layer, project structure

## Goal

Thiết lập nền tảng hạ tầng hoàn chỉnh: Docker multi-env, CI/CD, giao tiếp RPi↔Jetson, project scaffold.

## Prerequisite

Phase -1 (Specifications) complete — all 4 spec documents finalized.

## Spec References

- Communication: `docs/specs/communication-protocol.md`
- Safety: `docs/specs/safety-state-machine.md`
- Benchmarking: `docs/specs/benchmarking.md`

---

## Tasks

- [ ] **T0.1**: Tạo project scaffold (toàn bộ thư mục + `pyproject.toml` + `.gitignore`)
  → Verify: `tree` hiển thị đúng cấu trúc (match master plan file structure)

- [ ] **T0.2**: Tạo config files (`dev.yaml`, `test.yaml`, `prod.yaml`) với schema validation
  → Verify: `python -c "from config import load_config; load_config('dev')"`

- [ ] **T0.3**: Tạo `Dockerfile.dev` (NVIDIA base, hot-reload, TensorRT engine contract)
  → Verify: `docker build -f docker/Dockerfile.dev -t ctx-aware:dev .` thành công

- [ ] **T0.4**: Tạo `Dockerfile.test` (CI-optimized, pytest, ruff)
  → Verify: `docker build -f docker/Dockerfile.test -t ctx-aware:test .`

- [ ] **T0.5**: Tạo `Dockerfile.prod` (minimal, multi-stage build)
  - Image/model volume uses **TensorRT `.engine` artifacts only**
  - TensorRT runtime included for on-device `.engine` load/build validation
  - Volume mount for engine cache: `ctx-aware-engines:/app/models/engines`
  → Verify: `docker build -f docker/Dockerfile.prod -t ctx-aware:prod .`

- [ ] **T0.6**: Tạo `docker-compose.yml` với services: ai-pipeline, comm-layer, monitor, watchdog
  - Add volume: `ctx-aware-engines` for `.engine` cache
  → Verify: `docker-compose config` validates

- [ ] **T0.7**: Implement TCP/UDP comm layer — **MUST match `communication-protocol.md`**
  - `src/comm/protocol.py`: packet envelope, CRC-16, message type enum (§2, §3)
  - `src/comm/lidar_receiver.py`: TCP:9090 server, LIDAR_SCAN deserialization (§5)
  - `src/comm/command_sender.py`: TCP:9091 client, NAV_CMD serialization (§5)
  → Verify: `pytest tests/unit/test_protocol.py` — roundtrip, CRC, bounds checks

- [ ] **T0.8**: Implement health monitor — **MUST match `communication-protocol.md` §4 + `safety-state-machine.md` §5**
  - `src/comm/health_monitor.py`: heartbeat sender (500ms) + SOH receiver
  - Heartbeat timeout: 2s (4 missed beats)
  - Timeout ownership: RPi (see `safety-state-machine.md` §5)
  → Verify: `pytest tests/unit/test_heartbeat.py` — timeout detection within 2000±100ms

- [ ] **T0.9**: Tạo GitHub Actions workflows
  - `ci.yml`: lint (ruff) + unit tests + type check
  - `build.yml`: build Docker images (multi-arch)
  - `deploy.yml`: tag v*.*.* → push registry → Jetson pulls
  - **`benchmark-ci.yml`**: CPU regression checks — latency, RSS, output parity (per `benchmarking.md` §2)
  - **`benchmark-jetson.yml`**: manual trigger or self-hosted runner — FPS, GPU RAM, TRT build (per `benchmarking.md` §3)
  - `security.yml`: Trivy container scan + dependency audit
  → Verify: CI passes trên GitHub

- [ ] **T0.10**: Tạo `entrypoint.sh` — validate TensorRT `.engine` artifacts
  - Check Docker volume for existing `.engine` files
  - Compare engine metadata/hash (sha256) — skip rebuild if match
  - If engine build/load fails → log ERROR and enter documented degraded/safe mode
  - Script runs idempotent
  → Verify: First run validates/builds `.engine`. Second run skips. Delete `.engine` + fail build/load → service does not silently run CPU fallback

## Done When

- [ ] `docker-compose up` chạy thành công trên Jetson
- [ ] CI/CD pipeline green trên GitHub (6 workflows)
- [ ] RPi mock client gửi LiDAR data → Jetson nhận được
- [ ] Packet format matches `communication-protocol.md` §2 exactly
- [ ] Engine cache volume works across container restarts
- [ ] Engine-only degraded/safe failure path tested
