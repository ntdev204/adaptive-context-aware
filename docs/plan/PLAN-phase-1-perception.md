# Phase 1: Perception Layer

> Detection + Tracking + Sensor Fusion (RGB-D + LiDAR + IMU)

## Goal

Xây dựng perception pipeline chạy ≥25 FPS trên Jetson, output: tracked entities với 3D position, velocity, và pose features.

## Tasks

- [ ] **T1.1**: Implement `detector.py` — YOLOv8-s inference via TensorRT `.engine`
  → Verify: Detect persons trong test image, mAP check

- [ ] **T1.2**: Implement `depth_proc.py` — RGB-D depth map → 3D bounding box per detection
  → Verify: 3D position (x,y,z) chính xác ±0.3m so với ground truth

- [ ] **T1.3**: Implement `lidar_proc.py` — LiDAR scan processing, obstacle clustering
  → Verify: Cluster objects từ mock LiDAR scan data

- [ ] **T1.4**: Implement `tracker.py` — BoT-SORT multi-object tracking + depth association
  → Verify: Track IDs stable qua 100 frames, MOTA check

- [ ] **T1.5**: Implement `imu_fusion.py` — EKF ego-motion estimation
  → Verify: Ego-velocity estimate vs ground truth

- [ ] **T1.6**: Implement `sensor_fusion.py` — merge camera + LiDAR + IMU
  → Verify: Fused output có đủ fields (position_3d, velocity, heading)

- [ ] **T1.7**: Feature extractor per entity — appearance embedding (128d) + motion features
  → Verify: Embedding similarity test (same person = high sim)

- [ ] **T1.8**: Benchmark pipeline trên Jetson — latency, RAM, FPS
  → Verify: ≥25 FPS, GPU RAM <3GB cho perception alone

## Done When

- [ ] Full perception pipeline: camera frame → tracked entities với 3D features
- [ ] ≥25 FPS trên Jetson Orin Nano
- [ ] Unit tests all pass
