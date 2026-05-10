# Phase 3: Behavior & Decision

> Intent prediction + Anomaly detection + Navigation commands

## Goal

Từ fusion output → dự đoán intent (hướng di chuyển, hành vi), phát hiện bất thường, ra lệnh navigation.

## Tasks

- [ ] **T3.1**: Implement `intent_predictor.py` — multi-head output
  - Head 1: Movement direction (8 directions + stationary)
  - Head 2: Activity class (walking, running, standing, sitting, interacting, ...)
  - Head 3: Trajectory prediction (next 1-2s position)
  → Verify: Accuracy trên test data, latency <3ms

- [ ] **T3.2**: Implement `anomaly_detector.py` — multi-method
  - Statistical: deviation từ normal behavior distribution
  - Learned: autoencoder reconstruction error
  - Temporal: sudden change detection
  → Verify: Recall ≥80% trên synthetic anomaly cases

- [ ] **T3.3**: Implement `nav_commander.py` — context-aware navigation
  - Input: all entity intents + anomaly scores + robot goal
  - Output: velocity command (vx, vy, ω) cho mecanum drive
  → Verify: Robot avoids predicted collision paths

- [ ] **T3.4**: Scene context classifier — identify environment type
  - Classes: corridor, lobby, gate area, restaurant, open space
  → Verify: Classification accuracy trên test images

- [ ] **T3.5**: Integration test — full pipeline đến decision output
  → Verify: Given crowded scene → correct intents + anomaly flags + safe nav cmd

## Done When

- [ ] Intent prediction works cho ≥5 activity classes
- [ ] Anomaly detection recall ≥80%
- [ ] Navigation commands safe (no predicted collisions)
