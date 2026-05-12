# Báo cáo Validation: PHASE 1 (Perception Layer)

> Cập nhật: 2026-05-10
> Branch chạy validation: `phase/1-perception`
> Trạng thái: Hoàn thành một phần, còn 1 blocker môi trường Docker

---

## 1. Docker Validation

### 1.1 Dev Image
- **Build:** thành công với tag `ctx-aware:dev`
- **Smoke run:** thành công với lệnh `docker run --rm ctx-aware:dev python -m src.main`
- **Kết quả:** app boot bình thường và in `adaptive-context-aware booting in dev mode`

### 1.2 Test Image
- **Build:** thành công với tag `ctx-aware:test`
- **Run:** thất bại khi chạy `docker run --rm ctx-aware:test`
- **Root cause thực tế:** image test hiện build từ `python:3.10-slim` trên branch `phase/1-perception`, trong khi code đang dùng `StrEnum` ở [src/utils/enums.py](D:/utc/adaptive-context-aware/src/utils/enums.py), vốn cần Python 3.11 nếu không có backport.
- **Biểu hiện lỗi:** `ImportError: cannot import name 'StrEnum' from 'enum'`

### 1.3 Kết luận Docker
- **Dev Docker:** pass
- **Test Docker build:** pass
- **Test Docker runtime:** fail do mismatch Python runtime (`3.10`) và syntax/API code (`3.11`)

---

## 2. Benchmark Perception

### 2.1 Lệnh chạy
```powershell
python scripts/benchmark.py --device perception --frames 100
```

### 2.2 Kết quả thực tế
- **FPS mean:** `194.13`
- **Latency p50:** `5.12 ms`
- **Latency p95:** `6.91 ms`
- **Latency p99:** `10.03 ms`
- **Latency max:** `11.37 ms`
- **Peak RAM metric:** `256.0 MB`
- **Idle RAM metric:** `256.0 MB`
- **Frames measured:** `100`

### 2.3 Per-module latency
- **Detector p50:** `4.79 ms`
- **Detector p95:** `6.67 ms`
- **Tracker p50:** `0.00395 ms`
- **Tracker p95:** `0.00912 ms`
- **Fusion p50:** `0.00070 ms`
- **Fusion p95:** `0.00214 ms`

### 2.4 Threshold check
- **Target FPS ≥25:** pass
- **Target peak RAM <3072 MB:** pass

> Ghi chú: benchmark này là benchmark perception baseline hiện có trong repo, không phải Jetson hardware truth với detector engine thật end-to-end.

---

## 3. Phase 1 Status

### 3.1 Những gì đã được xác nhận
- Detector, depth, lidar, tracker, imu fusion, sensor fusion, feature extractor, và perception benchmark hook đều đã có unit tests.
- Perception benchmark baseline chạy được và vượt ngưỡng mục tiêu Phase 1 trên workload hiện có.
- Dev container khởi động được.

### 3.2 Blocker còn lại để đóng Phase 1 sạch hơn
- Docker runtime của branch `phase/1-perception` chưa đồng bộ sang Python 3.11.
- Vì vậy test image chạy thật trong container chưa pass.

---

## 4. Các bước tiếp theo cần làm (Next Steps)

### 1. Đồng bộ Python 3.11 vào `phase/1-perception`
- **Mục tiêu:** làm cho Docker runtime, CI, và code Phase 1 cùng một version Python.
- **Hướng dẫn thực hiện chi tiết:**
  - Merge hoặc cherry-pick thay đổi từ PR/runtime update đã vào `develop` sang `phase/1-perception`.
  - Kiểm tra lại:
    - `docker/Dockerfile.dev`
    - `docker/Dockerfile.test`
    - `docker/Dockerfile.prod`
    - `pyproject.toml`
    - `.github/workflows/*.yml`

### 2. Rerun Docker validation sau khi sync 3.11
- **Mục tiêu:** xác nhận `ctx-aware:test` pass trong container.
- **Hướng dẫn thực hiện chi tiết:**
  - Build lại:
    - `docker build -f docker/Dockerfile.dev -t ctx-aware:dev .`
    - `docker build -f docker/Dockerfile.test -t ctx-aware:test .`
  - Chạy lại:
    - `docker run --rm ctx-aware:dev python -m src.main`
    - `docker run --rm ctx-aware:test`

### 3. Nếu cần benchmark Jetson thật
- **Mục tiêu:** thay benchmark baseline hiện tại bằng số đo trên thiết bị Jetson.
- **Hướng dẫn thực hiện chi tiết:**
  - Chạy trên Jetson:
    - `python scripts/benchmark.py --device perception --frames 1000`
  - Ghi lại:
    - FPS mean/p95/p99
    - peak RAM thực tế
    - detector backend đang dùng (`engine`)

