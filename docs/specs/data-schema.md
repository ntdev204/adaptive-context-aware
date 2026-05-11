# Data Schema Specification

> Sensor formats, record structure, annotation schema, model I/O contracts

---

## 1. Sensor Data Formats

| Sensor | Resolution | Dtype | Size/frame | Rate |
|--------|-----------|-------|------------|------|
| RGB Camera | 640×480 | uint8 BGR | 921,600B (~900KB) | 30 Hz |
| Depth Map | 640×480 | float32 meters | 1,228,800B (~1.2MB) | 30 Hz |
| LiDAR 2D | N×2 (typ. 360) | float32 (angle_rad, dist_m) | ~2,880B | 10 Hz |
| IMU | 1 sample | struct (see below) | 48B | 100 Hz |

### IMU Sample Format

```python
imu_sample = {
    "accel":     [float32, float32, float32],  # m/s², XYZ
    "gyro":      [float32, float32, float32],  # rad/s, XYZ
    "quat":      [float32, float32, float32, float32],  # orientation XYZW
    "timestamp": uint64,  # microseconds since epoch
}
# Byte layout: accel(3×4=12) + gyro(3×4=12) + quat(4×4=16) + timestamp(8) = 48B
# struct format: "!3f 3f 4f Q" — verified: struct.calcsize() == 48
```

---

## 2. Record Format (HDF5)

All sensor data recorded into a single HDF5 file per session.

```
session_YYYYMMDD_HHMMSS.h5
├── /metadata
│   ├── session_id:        string
│   ├── start_time:        uint64 (µs epoch)
│   ├── duration_s:        float32
│   ├── robot_config:      JSON string
│   └── environment:       string (e.g., "corridor_floor2")
│
├── /rgb_frames
│   ├── data:              uint8 [N, 480, 640, 3]  (chunked, gzip)
│   └── timestamps:        uint64 [N]
│
├── /depth_frames
│   ├── data:              float32 [N, 480, 640]  (chunked, gzip)
│   └── timestamps:        uint64 [N]
│
├── /lidar_scans
│   ├── data:              float32 [N, max_points, 2]  (padded)
│   ├── num_points:        uint32 [N]
│   └── timestamps:        uint64 [N]
│
├── /imu
│   ├── accel:             float32 [M, 3]
│   ├── gyro:              float32 [M, 3]
│   ├── quat:              float32 [M, 4]
│   └── timestamps:        uint64 [M]
│
└── /annotations          (optional, added post-recording)
    ├── frame_annotations: JSON string [N]  (per RGB frame)
    └── scene_annotations: JSON string [N]
```

### Chunking & Compression

- RGB: chunks `(10, 480, 640, 3)`, gzip level 4
- Depth: chunks `(10, 480, 640)`, gzip level 4
- LiDAR: chunks `(100, max_points, 2)`, no compression
- IMU: chunks `(1000, *)`, no compression

---

## 3. Annotation Schema

### Per-Frame Annotation (JSON)

```json
{
  "frame_id": 0,
  "timestamp": 1715000000000000,
  "persons": [
    {
      "bbox": [120, 80, 50, 120],
      "track_id": 1,
      "activity": "WALKING",
      "intent_direction": "NORTH",
      "trajectory_pred": [[130, 75], [140, 70]],
      "is_anomaly": false,
      "confidence": 0.92
    }
  ],
  "scene": {
    "context": "CORRIDOR",
    "crowd_density": 0.15,
    "motion_entropy": 0.22,
    "anomaly_flag": false
  }
}
```

### JSON Schema (machine-parseable)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["frame_id", "timestamp", "persons", "scene"],
  "properties": {
    "frame_id": {"type": "integer", "minimum": 0},
    "timestamp": {"type": "integer"},
    "persons": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["bbox", "track_id", "activity"],
        "properties": {
          "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
          "track_id": {"type": "integer", "minimum": 0},
          "activity": {"type": "string", "enum": ["WALKING","RUNNING","STANDING","SITTING","INTERACTING","FALLING","LOITERING","FIGHTING","OTHER"]},
          "intent_direction": {"type": "string", "enum": ["NORTH","NE","EAST","SE","SOUTH","SW","WEST","NW","STATIONARY"]},
          "trajectory_pred": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
          "is_anomaly": {"type": "boolean"},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    },
    "scene": {
      "type": "object",
      "required": ["context", "crowd_density"],
      "properties": {
        "context": {"type": "string", "enum": ["CORRIDOR","LOBBY","GATE_AREA","RESTAURANT","OPEN_SPACE","UNKNOWN"]},
        "crowd_density": {"type": "number", "minimum": 0, "maximum": 1},
        "motion_entropy": {"type": "number", "minimum": 0, "maximum": 1},
        "anomaly_flag": {"type": "boolean"}
      }
    }
  }
}
```

---

## 4. Enums

### Activity

| Value | Description | Notes |
|-------|-------------|-------|
| `WALKING` | Normal walking pace | Most common |
| `RUNNING` | Fast movement | May trigger higher complexity |
| `STANDING` | Stationary, upright | |
| `SITTING` | Seated | |
| `INTERACTING` | Interacting with person/object | Social context |
| `FALLING` | Fall detected | **Anomaly** |
| `LOITERING` | Stationary too long | **Anomaly** (context-dependent) |
| `FIGHTING` | Aggressive interaction | **Anomaly** |
| `OTHER` | Unclassified | |

### Scene Context

| Value | Description |
|-------|-------------|
| `CORRIDOR` | Narrow passage, directional flow |
| `LOBBY` | Open area, multi-directional flow |
| `GATE_AREA` | Choke point, high density |
| `RESTAURANT` | Seated + moving people mixed |
| `OPEN_SPACE` | Large area, sparse |
| `UNKNOWN` | Cannot classify |

### Intent Direction (8-way + stationary)

`NORTH, NE, EAST, SE, SOUTH, SW, WEST, NW, STATIONARY`

---

## 5. Model I/O Contracts

| Model | Input Shape | Input Dtype | Output Shape | Output Dtype | Latency Target |
|-------|-------------|-------------|--------------|-------------|----------------|
| YOLOv11-s | `[1, 3, 480, 640]` | float32 | `[N, 6]` (x,y,w,h,conf,cls) | float32 | <8ms |
| BoT-SORT | detections + depth | mixed | `[M, 5]` (x,y,w,h,track_id) | float32/int | <3ms |
| Complexity Estimator | `[1, 36]` | float32 | `[1, 4]` (logits) | float32 | <1ms |
| GRU Pathway | `[B, T, 128]` | float32 | `[B, 64]` | float32 | <2ms |
| TCN Pathway | `[B, 128, T]` | float32 | `[B, 64]` | float32 | <3ms |
| Attention Pathway | `[B, N, 128]` | float32 | `[B, 128]` | float32 | <8ms |
| GNN Pathway | `[B, N, 128]` + adj `[N, N]` | float32 | `[B, 256]` | float32 | <15ms |
| Intent Predictor | `[B, D]` | float32 | 3 heads (direction, activity, traj) | float32 | <3ms |
| Anomaly Detector | `[B, D]` | float32 | `[B, 1]` (score) | float32 | <2ms |
| RL Policy (PPO) | `[1, 39]` | float32 | `[1, 4]` (action logits) | float32 | <1ms |

### Complexity Estimator Input Vector (36-dim)

```python
estimator_input = [
    crowd_density,       # 1 float
    motion_entropy,      # 1 float
    anomaly_score_prev,  # 1 float
    soh_budget,          # 1 float
    *scene_embedding,    # 32 floats
]
# Total: 36 dimensions
```

### RL Policy Input Vector (39-dim)

```python
rl_state = [
    crowd_density,          # 1
    motion_entropy,         # 1
    anomaly_probability,    # 1
    soh_budget,             # 1
    *scene_embedding,       # 32
    prev_action,            # 1 (scalar: action_id / num_actions, range [0, 0.75])
    time_since_critical,    # 1
    current_fps,            # 1
]
# Total: 39 dimensions
# Note: prev_action is a normalized scalar (0=FAST, 0.25=BALANCED, 0.5=ACCURATE, 0.75=EMERGENCY).
# NOT one-hot. This keeps the vector fixed at 39-dim regardless of action space changes.
```

---

## 6. Bootstrap Datasets

| Purpose | Source | Size | Location | Strategy |
|---------|--------|------|----------|----------|
| Detection smoke test | CCTV person subset | 20 images | `tests/fixtures/cctv_person_subset/` | **Extract locally** (gitignored) |
| Tracking sanity | MOT20 subset | 2 short clips | `tests/fixtures/mot20_subset/` | **Extract locally** (gitignored) |
| Quick image tests | Resized person images | 5 × 320×240 | `tests/fixtures/images/` | **Committed** (<1MB) |
| Anomaly test cases | Synthetic (hand-crafted) | 20 cases (JSON) | `tests/fixtures/anomaly_synthetic/` | **Committed** (<100KB) |
| RL routing scenarios | Synthetic (5 fixed scenarios) | 5 episodes (JSON) | `tests/fixtures/rl_scenarios/` | **Committed** (<50KB) |
| Sample HDF5 | Synthetic recording | 5s, all sensors | `tests/fixtures/sample_recording.h5` | **Generated** by script (~2MB) |
| Annotation samples | Hand-annotated | 5 frames (JSON) | `tests/fixtures/annotations/` | **Committed** (<10KB) |

> **Repo hygiene:** Only tiny fixtures (<10MB total) are committed. Larger bootstrap subsets are extracted locally from `data/fine_tuning/` via `scripts/download_fixtures.py` and gitignored.

### Upgrade Path

| Bootstrap → Production | Trigger |
|------------------------|---------|
| CCTV subset → Custom robot data | Phase 0.5/7: >500 annotated frames collected |
| MOT20 subset → Custom robot clips | Phase 7: >10 sequences recorded |
| Synthetic anomaly → UBnormal (optional) + robot data | Phase 7: robot anomaly clips available |
| Fixed RL scenarios → Environment traces | Phase 4: RL environment operational |

---

## 7. Verification

| Check | Command | Pass/Fail |
|-------|---------|-----------|
| Annotation JSON validates | `pytest tests/unit/test_data_schema.py -k json_schema` | All samples pass jsonschema |
| HDF5 read/write roundtrip | `pytest tests/unit/test_hdf5_io.py` | Record → replay identical |
| Model I/O shapes match | `pytest tests/unit/test_model_contracts.py` | All models accept/produce declared shapes |
| Enum completeness | `pytest tests/unit/test_data_schema.py -k enums` | All enum values documented and parseable |
| Fixture files exist | `pytest tests/unit/test_fixtures.py` | All paths in §6 resolve |
