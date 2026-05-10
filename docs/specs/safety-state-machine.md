# Safety State Machine Specification

> Jetson NEVER owns final motor stop. RPi or hardware layer is the safety authority.

---

## 1. Safety Philosophy

| Principle | Rule |
|-----------|------|
| **Jetson is advisory only** | Jetson can REQUEST state changes but never directly controls motors |
| **RPi owns motor stop (MVP)** | RPi monitors heartbeat, sends zero velocity, disables motor GPIO |
| **Hardware owns final stop (Production)** | Dedicated MCU/relay cuts power if RPi also fails |
| **No auto-recovery from ESTOP** | Manual reset required — prevents oscillation |
| **Fail-safe default** | Loss of signal = stop. Silence = danger |

---

## 2. Two-Tier Safety Model

### Tier 1: MVP (Software Watchdog)

```
Jetson → [heartbeat TCP:9093] → RPi → [GPIO] → Motor Controller

Safety chain:
  Jetson AI pipeline
       │ heartbeat (500ms)
       ▼
  RPi watchdog process
       │ monitors heartbeat
       │ timeout = 2s
       ▼
  RPi motor control
       │ zero velocity + GPIO disable
       ▼
  Motor Controller (H-bridge/ESC)
```

**Owner of motor stop: RPi**
- RPi runs a dedicated watchdog process (separate from main control loop)
- Watchdog is highest priority process on RPi
- If watchdog itself crashes: systemd auto-restarts within 1s

### Tier 2: Production (Hardware Watchdog)

```
Jetson → [heartbeat] → RPi → [heartbeat] → MCU/Relay → Motor Power

Safety chain:
  RPi sends heartbeat to hardware MCU (e.g., Arduino Nano / ATtiny85)
  MCU expects heartbeat every 1s
  If 3s without heartbeat → relay cuts motor power line
  Physical E-stop button wired directly to relay (bypasses all software)
```

**Owner of final stop: Hardware MCU/Relay**
- Independent of all software on both RPi and Jetson
- Simple firmware: receive heartbeat → reset timer → if timer expires → open relay
- Physical E-stop button is hardwired to relay — zero software in path

---

## 3. State Machine

### States

| State | Code | Description | Motor Status | AI Pipeline |
|-------|------|-------------|--------------|-------------|
| `NORMAL` | 0 | All systems healthy | Full speed allowed | Full pipeline |
| `DEGRADED` | 1 | Partial failure, safe to continue | Reduced max speed | Fallback pathway |
| `ESTOP` | 2 | Emergency stop active | **All motors stopped** | Logging only |
| `RECOVERY` | 3 | Self-test in progress | Motors stopped | Diagnostics running |

### State Diagram

```
                    ┌──────────┐
           ┌───────│  NORMAL  │◄──────── manual reset confirmed
           │       │  (0)     │          + self-test pass
           │       └────┬─────┘          (from RECOVERY)
           │            │
           │    GPU hot / sensor fail
           │    SoH < 0.3
           │            │
           │            ▼
           │       ┌──────────┐
           │       │ DEGRADED │  RPi: reduce max speed
           │       │  (1)     │  Jetson: GRU-only mode
           │       └────┬─────┘
           │            │
           │    heartbeat loss / anomaly CRITICAL
           │    manual button / further degradation
           │            │
           ▼            ▼
      ┌──────────────────────┐
      │       ESTOP (2)      │  RPi: zero velocity → motor GPIO disable
      │  (all motors stop)   │  Jetson: stop pipeline, log, alert
      └──────────┬───────────┘
                 │
         manual reset button/command
         (operator confirmation required)
                 │
                 ▼
           ┌──────────┐
           │ RECOVERY │  System runs self-test suite
           │  (3)     │  Motors remain stopped
           └──────────┘
              │     │
         pass │     │ fail
              ▼     ▼
          NORMAL   ESTOP
```

---

## 4. Transition Rules

| # | From | To | Trigger | Owner | Action | Timing |
|---|------|----|---------|-------|--------|--------|
| T1 | NORMAL | DEGRADED | GPU >80°C | Jetson (advisory) | Jetson sends `STATUS_UPDATE(DEGRADED, GPU_OVERHEAT)`. RPi ACKs, reduces max speed to 50% | — |
| T2 | NORMAL | DEGRADED | Camera failure | Jetson (advisory) | Jetson sends `STATUS_UPDATE(DEGRADED, CAMERA_FAIL)`. RPi ACKs, switches to LiDAR-only nav | — |
| T3 | NORMAL | DEGRADED | SoH budget < 0.3 | Jetson (advisory) | Jetson sends `STATUS_UPDATE(DEGRADED, SOH_LOW)`. RPi ACKs, reduces complexity | — |
| T4 | NORMAL | ESTOP | Jetson heartbeat lost | **RPi (owner)** | RPi detects 4 missed heartbeats → zero velocity → motor GPIO disable | **2s** |
| T5 | NORMAL | ESTOP | Manual E-stop button | **Hardware (GPIO)** | Interrupt-driven, immediate motor kill | **<1ms** |
| T6 | NORMAL | ESTOP | Critical anomaly | Jetson (advisory) | Jetson sends `ESTOP(ANOMALY_CRITICAL)`. RPi executes motor stop | <100ms |
| T7 | DEGRADED | NORMAL | Condition resolved + 10s stable | Jetson (advisory) | Jetson sends `STATUS_UPDATE(NORMAL, CONDITION_RESOLVED)`. RPi resumes full speed | 10s hysteresis |
| T8 | DEGRADED | ESTOP | Heartbeat loss | **RPi (owner)** | Same as T4 | **2s** |
| T9 | DEGRADED | ESTOP | Further degradation (multiple faults) | **RPi (owner)** | RPi determines too many faults → ESTOP | — |
| T10 | ESTOP | RECOVERY | Manual reset (button or confirmed command) | **Operator** | System enters self-test mode. Motors remain stopped | — |
| T11 | RECOVERY | NORMAL | All self-tests pass | System | Resume normal operation | 5s test window |
| T12 | RECOVERY | ESTOP | Any self-test fails | System | Back to ESTOP, log failure reason | — |
| T13 | ANY | ESTOP | Physical E-stop button | **Hardware** | Immediate motor power cut, no software involved | **0ms** |

### Invalid Transitions (must be rejected)

| From | To | Why |
|------|----|-----|
| ESTOP | NORMAL | Must go through RECOVERY first |
| ESTOP | DEGRADED | Must go through RECOVERY first |
| RECOVERY | DEGRADED | Recovery either succeeds (NORMAL) or fails (ESTOP) |
| NORMAL | RECOVERY | Recovery only entered from ESTOP |

---

## 5. Heartbeat Contract

### Jetson → RPi (TCP:9093)

| Field | Value |
|-------|-------|
| Interval | 500ms |
| Payload | `{state: u8, pipeline_fps: f32, gpu_temp_c: u8}` (8B) |
| Timeout | 2000ms (4 consecutive misses) |
| Monitor owner | RPi watchdog process |
| Monitor frequency | Check every 100ms |
| On timeout | Transition to ESTOP |

### RPi Watchdog Implementation Requirements

```python
# Pseudocode for RPi watchdog (separate process, highest priority)
class HeartbeatWatchdog:
    INTERVAL_MS = 500
    TIMEOUT_MS = 2000
    CHECK_EVERY_MS = 100

    def monitor(self):
        while True:
            elapsed = now() - self.last_heartbeat_time
            if elapsed > self.TIMEOUT_MS:
                self.trigger_estop(reason=HEARTBEAT_TIMEOUT)
            sleep(self.CHECK_EVERY_MS)

    def trigger_estop(self, reason):
        # 1. Send zero velocity command to motor controller
        self.motor.set_velocity(0, 0, 0)
        # 2. Disable motor GPIO (fail-safe)
        self.gpio.disable_motors()
        # 3. Update state
        self.state = SafetyState.ESTOP
        # 4. Log
        self.logger.critical(f"ESTOP triggered: {reason}")
```

---

## 6. AI Pipeline Failure Scenarios

| Scenario | What Happens | Who Acts |
|----------|-------------|----------|
| Jetson AI pipeline crashes, comm alive | Jetson heartbeat stops → RPi detects timeout → ESTOP | RPi |
| Jetson AI produces bad output | Jetson sends `STATUS_UPDATE(DEGRADED)` → RPi reduces speed | Jetson (advisory), RPi (executes) |
| Jetson AI detects critical anomaly | Jetson sends `ESTOP(ANOMALY_CRITICAL)` → RPi stops motors | Jetson (advisory), RPi (executes) |
| Jetson Docker container restarts | Heartbeat gap → if gap exceeds 2s timeout: enter ESTOP (manual reset required). If gap <2s: RPi sees reconnection, system remains in current state (ESTOP never entered, so "no auto-recovery" rule is not violated) | RPi |
| Jetson loses camera | Jetson sends `STATUS_UPDATE(DEGRADED, CAMERA_FAIL)` → LiDAR-only mode | Jetson (advisory) |
| Jetson GPU OOM | Jetson sends `STATUS_UPDATE(DEGRADED, SOH_LOW)` → GRU-only pathway | Jetson (advisory) |

---

## 7. RPi Failure Scenarios

### MVP (Software Only)

| Scenario | What Happens | Mitigation |
|----------|-------------|------------|
| RPi watchdog process crashes | systemd restarts watchdog within 1s. If gap <2s: likely no ESTOP. If >2s: motors were already being commanded by main loop | systemd `Restart=always`, `RestartSec=0.5` |
| RPi loses power | Motors lose command signal. Motor controller behavior depends on H-bridge: some coast, some brake | **Unacceptable for production** → need Tier 2 |
| RPi OS crashes | Same as power loss | Same |
| RPi network fails (Wi-Fi to Jetson) | Wired TCP connection unaffected. If using Wi-Fi: heartbeat loss → ESTOP | Use **wired Ethernet** between RPi and Jetson |

### Production (Hardware Watchdog)

| Scenario | What Happens | Mitigation |
|----------|-------------|------------|
| RPi loses power | Hardware MCU detects RPi heartbeat loss (3s) → relay opens → motor power cut | MCU firmware |
| RPi OS crashes | Same as power loss | Same |
| MCU fails | Relay defaults to **open** (normally-open relay wired so power-off = motor stop) | Fail-safe relay wiring |

---

## 8. Recovery Protocol

### From ESTOP → RECOVERY → NORMAL

1. **Operator presses reset** (physical button or software command with confirmation)
2. System enters `RECOVERY` state
3. Self-test suite runs (5s window):
   - [ ] Jetson heartbeat active?
   - [ ] RPi motor control responsive?
   - [ ] Camera streaming?
   - [ ] LiDAR scanning?
   - [ ] GPU temperature <75°C?
   - [ ] RAM usage <80%?
4. If ALL pass → transition to `NORMAL`
5. If ANY fail → back to `ESTOP`, log which test failed

### No Auto-Recovery Rule

- ESTOP state **persists indefinitely** without manual intervention
- This prevents: crash → auto-restart → crash → auto-restart loops
- Rationale: robot in physical environment, unattended restart is dangerous

---

## 9. Motor Controller Behavior

### Commanded Stop vs Power Loss

| Event | Motor Behavior | Source |
|-------|---------------|--------|
| Zero velocity command | Controlled deceleration, motors brake | RPi software |
| GPIO motor disable | Motor driver disabled, motors coast briefly then stop | RPi GPIO |
| Relay power cut (Tier 2) | Immediate power loss, motors coast | Hardware MCU |
| Physical E-stop | Immediate power cut via relay | Hardwired button |

### Motor Controller Requirements

- Must accept zero velocity as valid command (not reject as "no command")
- GPIO disable pin must be **active-low** (default = disabled = safe)
- On startup: motors disabled until explicit enable command after NORMAL state confirmed

---

## 10. Implementation Reference

| Component | File | Spec Section |
|-----------|------|-------------|
| Safety FSM class | `src/safety/state_machine.py` | §3, §4 |
| E-stop handler | `src/safety/estop.py` | §5, §6 |
| Graceful degradation | `src/safety/graceful_degrade.py` | T1-T3, T7 |
| Watchdog | `src/safety/watchdog.py` | §5 |
| Recovery self-test | `src/safety/recovery.py` | §8 |

---

## 11. Verification

| Check | Command | Pass/Fail |
|-------|---------|-----------|
| All valid transitions work | `pytest tests/unit/test_safety_fsm.py -k valid_transitions` | All 13 transitions succeed |
| Invalid transitions rejected | `pytest tests/unit/test_safety_fsm.py -k invalid_transitions` | `InvalidTransitionError` for all invalid combos |
| Heartbeat timeout → ESTOP | `pytest tests/integration/test_heartbeat_estop.py` | ESTOP fires within 2000±100ms |
| ESTOP persists without reset | `pytest tests/unit/test_estop_recovery.py -k no_auto_recovery` | State remains ESTOP after 30s |
| Recovery self-test pass → NORMAL | `pytest tests/unit/test_estop_recovery.py -k recovery_pass` | Transition succeeds |
| Recovery self-test fail → ESTOP | `pytest tests/unit/test_estop_recovery.py -k recovery_fail` | Back to ESTOP |
| Motor stop ownership = RPi | Code review: `estop.py` | Jetson never directly controls motors |
| Physical E-stop bypasses software | Hardware test (manual) | Button → motors stop <1ms |
