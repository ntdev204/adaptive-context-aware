# Báo cáo Tiến trình: PHASE 1 (Perception Layer)

> Cập nhật: 2026-05-10
> Trạng thái: Hoàn thành

---

## PHÂN HỆ I: PHASE 1.0 (Perception Core)

### 1. Detection
- **Detector runtime baseline:** chuyển từ synthetic-only sang engine-first runtime cho Jetson; image bootstrap tự kéo model thật, export artifact trung gian và build TensorRT engine khi container start.
- **Contract bảo toàn:** input `[1, 3, 480, 640]`, output detections `[N, 6]`, downstream không phải đổi interface khi thay backend.

### 2. Depth + LiDAR + Tracking
- **Depth projection:** từ detection + depth map suy ra 3D bbox theo intrinsics, trả về `x_m/y_m/z_m/width_m/height_m`.
- **LiDAR clustering:** lọc point invalid, chuyển polar → Cartesian, cluster obstacle theo continuity trong scan.
- **Tracker baseline:** ghép detection + depth bằng `IoU + depth gate`, giữ `track_id`, tính `velocity_3d`, loại track stale.

### 3. IMU + Sensor Fusion
- **IMU baseline:** tích phân gia tốc để ước lượng ego-velocity, suy yaw từ quaternion.
- **Fusion baseline:** gộp `TrackState + EgoMotionState + LidarCluster` thành entity record thống nhất cho downstream.

---

## PHÂN HỆ II: PHASE 1.1 (Entity Features + Benchmark)

### 1. Feature Extraction
- **Appearance embedding 128d:** tạo embedding deterministic từ bbox + position + velocity + heading + obstacle context để downstream có thể dùng ngay.
- **Similarity contract:** cùng entity cho similarity cao, khác entity cho similarity thấp hơn, đủ để thay bằng CNN/metric learning model thật sau này.

### 2. Benchmark Pipeline
- **Perception benchmark riêng:** thêm runner cho perception-only, đo `fps`, `latency_ms`, `gpu_ram_mb`, và thresholds `min_fps=25`, `max_peak_rss_mb=3072`.
- **CI baseline vẫn synthetic có chủ đích:** benchmark CI có warning rõ ràng để không nhầm với số đo Jetson thật.

---

## PHÂN HỆ III: Phase 1 Infrastructure Support

### 1. Codebase Support
- **Package scaffold:** thêm `src/perception/*`, test suite tương ứng, và update `pyproject.toml`/`ruff` để giữ code hygiene.
- **Benchmark helper:** `scripts/benchmark.py` hỗ trợ `--device perception` cho pipeline report; `scripts/update_ci_baselines.py` tiếp tục dùng cho CI regression placeholder.

### 2. Validation
- **Unit tests:** detector, depth, lidar, tracker, imu fusion, sensor fusion, feature extractor, benchmark helper đều có test riêng.
- **Workflow:** mọi task Phase 1 đều đi theo branch task → merge vào `phase/1-perception` → cleanup local + remote branch.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Chốt review Phase 1 và chuẩn bị Phase 2
- **Mục tiêu:** khóa trạng thái Phase 1 đã hoàn tất, không để task branch tồn đọng.
- **Hướng dẫn thực hiện chi tiết:**
  - Kiểm tra `git status --short --branch` trên `phase/1-perception`.
  - Nếu sạch, giữ branch phase làm baseline.
  - Đọc `docs/plan/PLAN-phase-2-adaptive-core.md` để xác định task Phase 2 đầu tiên.
  - Lệnh test tham chiếu: `python -m ruff check src tests scripts` và `python -m pytest tests/unit -q`.

### 2. Bắt đầu Phase 2 theo branch đúng ngữ nghĩa
- **Mục tiêu:** tạo `phase/2-adaptive-core` từ `develop` hoặc theo chiến lược phase bạn chốt.
- **Hướng dẫn thực hiện chi tiết:**
  - Chuyển sang branch gốc đã chốt cho phase mới.
  - Tạo branch phase mới, sau đó tách branch task đầu tiên cho `T2.1`.
  - Cập nhật file plan nếu còn nhãn/contract phải đồng bộ giữa Phase 1 và Phase 2.

### 3. Nếu cần nâng benchmark perception lên số thật
- **Mục tiêu:** thay benchmark synthetic bằng số đo Jetson thật cho pipeline perception-only.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `scripts/benchmark.py`.
  - Thay `run_perception_benchmark()` bằng measurement backend thật trên Jetson.
  - Chạy lại `python scripts/benchmark.py --device perception --frames 1000`.

