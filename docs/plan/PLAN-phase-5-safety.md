# Phase 5: Safety & Monitoring

> Fault tolerance, safety state machine, watchdog, E-stop, metrics, logging

## Goal

Hệ thống không được chết — graceful degradation, auto-recovery, comprehensive monitoring. **Jetson is advisory only. RPi or hardware layer owns final motor stop.**

## Spec Reference

- **Primary:** `docs/specs/safety-state-machine.md` — all states, transitions, ownership
- **Communication:** `docs/specs/communication-protocol.md` — heartbeat format, ESTOP packet

---

## Tasks

- [ ] **T5.1**: Implement `watchdog.py` — container + process health checks
  - Heartbeat mỗi 5s, auto-restart nếu fail
  → Verify: Kill process → auto-restart trong <5s

- [ ] **T5.2**: Implement `graceful_degrade.py` — degradation levels
  - GPU OOM → GRU-only mode
  - Camera fail → LiDAR-only obstacle avoidance
  - High temp (>80°C) → reduce FPS target
  - Implements transitions T1, T2, T3, T7 from `safety-state-machine.md` §4
  → Verify: Simulate mỗi failure → correct fallback + STATUS_UPDATE sent

- [ ] **T5.3**: Implement `estop.py` — emergency stop system
  - RPi heartbeat lost >2s → stop motors
  - Comm loss → safe stop
  - Manual E-stop signal handling
  - **Motor stop ownership: RPi only** (see `safety-state-machine.md` §1)
  - Jetson sends ESTOP advisory; RPi executes motor stop
  → Verify: Disconnect RPi mock → motors stop cmd sent trong <2s

- [ ] **T5.4**: Implement `metrics.py` — system metrics collection
  - FPS, latency per module, GPU/CPU/RAM usage, pathway selection distribution
  → Verify: Metrics exportable dạng JSON/Prometheus

- [ ] **T5.5**: Implement `logger.py` — structured logging with levels per env
  - dev=DEBUG, test=INFO, prod=WARNING
  - Log rotation, max file size
  → Verify: Logs written correctly, rotation works

- [ ] **T5.6**: systemd service file cho auto-start docker-compose on boot
  → Verify: Reboot Jetson → containers auto-start

- [ ] **T5.7**: Implement `state_machine.py` — Safety FSM
  - 4 states: NORMAL, DEGRADED, ESTOP, RECOVERY
  - All 13 transition rules from `safety-state-machine.md` §4
  - Invalid transition rejection (ESTOP→NORMAL, ESTOP→DEGRADED, etc.)
  - State persistence: ESTOP requires manual reset (no auto-recovery)
  → Verify: `pytest tests/unit/test_safety_fsm.py`
    - All valid transitions succeed
    - All invalid transitions raise `InvalidTransitionError`
    - Timeout enforcement correct

- [ ] **T5.8**: Implement `recovery.py` — self-test suite for RECOVERY state
  - Check: Jetson heartbeat, RPi motor control, camera, LiDAR, GPU temp, RAM
  - 5s test window
  - Pass → NORMAL, fail → ESTOP
  → Verify: `pytest tests/unit/test_estop_recovery.py`
    - Recovery pass → NORMAL
    - Recovery fail → back to ESTOP

- [ ] **T5.9**: Integration test — heartbeat loss → ESTOP → manual recovery → NORMAL
  → Verify: `pytest tests/integration/test_heartbeat_estop.py`
    - Simulated heartbeat loss → ESTOP within 2s
    - Manual reset → RECOVERY → self-test → NORMAL

## Done When

- [ ] Container crash → auto-restart <5s
- [ ] GPU overheat → graceful degradation (STATUS_UPDATE sent)
- [ ] Comm loss → E-stop <2s
- [ ] Safety FSM implements all transitions from `safety-state-machine.md` §4
- [ ] Heartbeat timeout triggers ESTOP within 2s (RPi-owned)
- [ ] ESTOP persists without manual reset (no auto-recovery)
- [ ] Full metrics dashboard available
- [ ] Motor stop ownership = RPi (code review verified)
