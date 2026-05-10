# Communication Protocol Specification

> RPi 4 ↔ Jetson Orin Nano — TCP/UDP, binary packets, heartbeat-driven safety

---

## 1. Transport Map

| Link | Protocol | Port | Direction | Rationale |
|------|----------|------|-----------|-----------|
| LiDAR scan data | TCP | 9090 | RPi → Jetson | Reliable delivery; scan integrity critical |
| Navigation commands | TCP | 9091 | Jetson → RPi | Motor commands must arrive |
| SoH telemetry | UDP | 9092 | RPi → Jetson | Best-effort; latest value wins |
| Heartbeat + E-Stop | TCP | 9093 | Bidirectional | Reliability for safety-critical signals |

---

## 2. Packet Format

```
┌──────────┬──────────┬──────┬────────────┬─────────────┬─────────┬────────┐
│ magic    │ msg_type │ seq  │ timestamp  │ payload_len │ payload │ crc16  │
│ 2B       │ 1B       │ 4B   │ 8B (µs)   │ 4B          │ N bytes │ 2B     │
│ 0xCA 0xFE│          │ u32  │ u64        │ u32         │ varies  │ CCITT  │
└──────────┴──────────┴──────┴────────────┴─────────────┴─────────┴────────┘
Total overhead: 21 bytes + payload
```

### Python struct

```python
HEADER_FORMAT = "!2s B I Q I"  # magic(2) + msg_type(1) + seq(4) + timestamp(8) + payload_len(4)
HEADER_SIZE = 19
CRC_SIZE = 2
```

---

## 3. Message Types

```python
from enum import IntEnum

class MsgType(IntEnum):
    LIDAR_SCAN     = 0x01
    NAV_CMD        = 0x02
    HEARTBEAT      = 0x03
    ESTOP          = 0x04
    SOH_TELEMETRY  = 0x05
    ACK            = 0x06
    STATUS_UPDATE  = 0x07
```

| Value | Name | Port | Direction | Payload Size |
|-------|------|------|-----------|--------------|
| `0x01` | `LIDAR_SCAN` | 9090 | RPi → Jetson | Variable (N×8B) |
| `0x02` | `NAV_CMD` | 9091 | Jetson → RPi | 20B |
| `0x03` | `HEARTBEAT` | 9093 | Bidirectional | 8B |
| `0x04` | `ESTOP` | 9093 | Jetson → RPi | 4B |
| `0x05` | `SOH_TELEMETRY` | 9092 | RPi → Jetson | 32B |
| `0x06` | `ACK` | 9093 | Bidirectional | 8B |
| `0x07` | `STATUS_UPDATE` | 9093 | Jetson → RPi | 4B |

---

## 4. Heartbeat Protocol

### Jetson → RPi

- **Transport:** TCP:9093
- **Interval:** 500ms (2 Hz)
- **Payload:** `{state: u8, pipeline_fps: f32, gpu_temp_c: u8}` (8B)

### RPi → Jetson SoH

- **Transport:** UDP:9092
- **Interval:** 1000ms (1 Hz)
- **Payload:** SOH telemetry (§5)

### Timeout Detection (RPi-owned)

```
RPi monitors Jetson heartbeat on TCP:9093:
  Expected interval: 500ms
  Timeout: 2000ms (4 consecutive misses)
  Check frequency: every 100ms
  On timeout → ESTOP (see safety-state-machine.md)
```

### Connection Management

- RPi: TCP **server** on 9093 (listens)
- Jetson: TCP **client** (connects to RPi)
- Connection drop: RPi starts 2s timeout immediately
- Reconnect: exponential backoff 100ms→200ms→400ms→800ms→1s max

---

## 5. Command Schemas

### NAV_CMD (0x02) — Jetson → RPi, TCP:9091

```python
NAV_CMD_FORMAT = "!f f f I H H"  # 20 bytes — verified: struct.calcsize() == 20
payload = {
    "vx":       float32,  # m/s, forward(+)/backward(-), range [-1.0, 1.0]
    "vy":       float32,  # m/s, left(+)/right(-), range [-1.0, 1.0]
    "omega":    float32,  # rad/s, CCW(+), range [-2.0, 2.0]
    "cmd_seq":  uint32,   # dedup sequence
    "flags":    uint16,   # reserved for future use (e.g., motion mode)
    "reserved": uint16,   # padding to 20B
}
# Byte layout: f(4) + f(4) + f(4) + I(4) + H(2) + H(2) = 20
```

### ESTOP (0x04) — Jetson → RPi, TCP:9093

```python
ESTOP_FORMAT = "!B B H"  # 4 bytes
payload = {
    "reason":   uint8,   # EStopReason enum
    "source":   uint8,   # EStopSource enum
    "reserved": uint16,
}

class EStopReason(IntEnum):
    HEARTBEAT_TIMEOUT = 0x01
    ANOMALY_CRITICAL  = 0x02
    MANUAL_BUTTON     = 0x03
    SYSTEM_FAULT      = 0x04
    THERMAL_CRITICAL  = 0x05

class EStopSource(IntEnum):
    JETSON_AI    = 0x01
    RPI_WATCHDOG = 0x02
    HARDWARE     = 0x03
    OPERATOR     = 0x04
```

### LIDAR_SCAN (0x01) — RPi → Jetson, TCP:9090

```python
# Header: num_points (u32, 4B) + points (N × 2 × f32)
# Typical: 360 points = 2884B payload, 10 Hz
payload = {
    "num_points": uint32,
    "points":     [(float32, float32)] * N,  # (angle_rad, distance_m)
}
```

### SOH_TELEMETRY (0x05) — RPi → Jetson, UDP:9092

```python
SOH_FORMAT = "!f f f f f B B H I I"  # 32 bytes — verified: struct.calcsize() == 32
payload = {
    "cpu_temp_c":      float32,
    "cpu_util_pct":    float32,
    "ram_used_mb":     float32,
    "battery_v":       float32,  # 0 if wall-powered
    "motor_current_a": float32,
    "lidar_ok":        uint8,    # 1=healthy, 0=fault
    "motor_ok":        uint8,
    "reserved":        uint16,
    "uptime_s":        uint32,   # RPi uptime in seconds (useful for diagnostics)
    "reserved2":       uint32,   # padding to 32B
}
# Byte layout: 5×f(20) + B(1) + B(1) + H(2) + I(4) + I(4) = 32
```

### HEARTBEAT (0x03) — Bidirectional, TCP:9093

```python
HEARTBEAT_FORMAT = "!B f B H"  # 8 bytes — verified: struct.calcsize() == 8
payload = {
    "state":        uint8,    # SafetyState enum {NORMAL=0, DEGRADED=1, ESTOP=2, RECOVERY=3}
    "pipeline_fps": float32,  # Jetson→RPi only, 0 for RPi→Jetson
    "gpu_temp_c":   uint8,    # Jetson→RPi only
    "reserved":     uint16,
}
# Byte layout: B(1) + f(4) + B(1) + H(2) = 8
```

### STATUS_UPDATE (0x07) — Jetson → RPi, TCP:9093

```python
STATUS_FORMAT = "!B B H"  # 4 bytes
payload = {
    "new_state": uint8,   # SafetyState requested
    "reason":    uint8,   # StatusChangeReason enum
    "reserved":  uint16,
}

class StatusChangeReason(IntEnum):
    GPU_OVERHEAT       = 0x01
    CAMERA_FAIL        = 0x02
    SOH_LOW            = 0x03
    CONDITION_RESOLVED = 0x04
    SELF_TEST_PASS     = 0x05
    SELF_TEST_FAIL     = 0x06
```

### ACK (0x06) — Bidirectional, TCP:9093

```python
ACK_FORMAT = "!B I B H"  # 8 bytes — verified: struct.calcsize() == 8
payload = {
    "ack_msg_type": uint8,   # which msg type acknowledged
    "ack_seq":      uint32,  # seq being acknowledged
    "status":       uint8,   # 0=OK, 1=REJECTED, 2=ERROR
    "reserved":     uint16,
}
# Byte layout: B(1) + I(4) + B(1) + H(2) = 8
```

---

## 6. Retry & Timeout Policy

| Transport | Strategy | Details |
|-----------|----------|---------|
| TCP (LiDAR, NAV_CMD) | TCP handles retransmit | App-level: no retry needed |
| TCP (ESTOP) | Send + verify | 3 retries × 200ms backoff. No ACK after 3 → RPi already ESTOP from heartbeat timeout |
| TCP (Heartbeat) | No retry | Absence IS the signal |
| UDP (SoH) | Fire-and-forget | Latest wins; stale data detected by timestamp |

### Latency Budget

| Message | Target | Notes |
|---------|--------|-------|
| ESTOP delivery | < 2ms | Highest priority |
| NAV_CMD delivery | < 2ms | 20B payload (4B padding for alignment) |
| Heartbeat roundtrip | < 5ms | 8B payload |
| LiDAR scan delivery | < 5ms | ~3KB payload |
| SoH telemetry | < 10ms | Best-effort |

---

## 7. Serialization

**MVP:** Python `struct.pack` / `struct.unpack`, network byte order (`!`). Zero dependencies, predictable performance.

**Future:** Migrate to Protocol Buffers if >10 message types or nested structures. Keep binary envelope (magic+CRC) around protobuf payload.

---

## 8. Sequence Diagrams

### Normal Operation Loop

```
Jetson                                RPi
  │◄──── LIDAR_SCAN (TCP:9090) ───────│  10 Hz
  │  [AI Pipeline processes]           │
  │──── NAV_CMD (TCP:9091) ──────────►│  10-20 Hz
  │──── HEARTBEAT (TCP:9093) ────────►│  2 Hz
  │◄──── SOH_TELEMETRY (UDP:9092) ────│  1 Hz
```

### Heartbeat Loss → E-Stop

```
Jetson                                RPi
  │──── HEARTBEAT ───────────────────►│  t=0ms
  │  [Jetson crashes]                  │
  │  ✗  (no heartbeat)                │  t=500ms  — miss #1
  │  ✗                                │  t=1000ms — miss #2
  │  ✗                                │  t=1500ms — miss #3
  │  ✗                                │  t=2000ms — miss #4 → ESTOP
  │                                    │  → zero velocity → motor disable
  │  [Jetson recovers, reconnects]     │
  │──── HEARTBEAT ───────────────────►│  ESTOP persists, manual reset required
```

### Manual E-Stop (Physical Button)

```
  [Operator presses button]           RPi
                                       │  GPIO interrupt (t=0ms)
                                       │  → motor power cut (<1ms)
                                       │  → state = ESTOP
Jetson                                 │
  │◄──── ESTOP (TCP:9093) ────────────│  inform Jetson
  │──── ACK ─────────────────────────►│
```

---

## 9. Implementation Reference

| Component | File | Spec Section |
|-----------|------|--------------|
| Packet envelope | `src/comm/protocol.py` | §2 |
| Message types | `src/comm/protocol.py` | §3 |
| LiDAR receiver | `src/comm/lidar_receiver.py` | §5 LIDAR_SCAN |
| Command sender | `src/comm/command_sender.py` | §5 NAV_CMD |
| Health monitor | `src/comm/health_monitor.py` | §4, §5 SOH |
| E-stop handler | `src/safety/estop.py` | §5 ESTOP, §8 |

## Verification

| Check | Command | Pass/Fail |
|-------|---------|-----------|
| Roundtrip serialize/deserialize | `pytest tests/unit/test_protocol.py -k roundtrip` | Zero data loss |
| Heartbeat timeout at 2s | `pytest tests/unit/test_heartbeat.py -k timeout` | Within 2000±100ms |
| CRC rejects corruption | `pytest tests/unit/test_protocol.py -k crc` | `InvalidPacketError` |
| NAV_CMD bounds | `pytest tests/unit/test_protocol.py -k nav_cmd_bounds` | Out-of-range rejected |
