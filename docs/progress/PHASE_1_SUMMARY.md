# Báo cáo Tiến trình: PHASE 1 (Perception Layer)

> Cập nhật: 2026-05-11
> Trạng thái: Đang tiến hành

---

## PHÂN HỆ I: PHASE 1.0 (Perception Core)

### 1. Detection + Tracking + Sensor Fusion
- **Detector chuyển sang engine-first runtime:** pipeline mặc định dùng backend `engine`; nếu thiếu file engine sẽ raise `FileNotFoundError`, nếu engine có lỗi load/chạy sẽ raise `TensorRTInferenceUnavailableError`, còn đường happy path dùng `TensorRTEngineRunner` và postprocess output về contract `[N, 6]`.
- **Depth/LiDAR/Tracker đã chốt baseline tính toán:** depth projection trả về thông tin 3D bbox theo intrinsics; LiDAR xử lý scan và clustering ổn định; tracker giữ `track_id`, kết hợp `IoU + depth gate`, loại stale tracks.
- **IMU và sensor fusion hoàn tất mặt interface:** IMU update cho ego-motion và fusion hợp nhất track + motion + lidar thành entity record thống nhất cho downstream.

## PHÂN HỆ II: PHASE 1.1 (Entity Features + Benchmark)

### 1. Entity Feature Layer + Benchmark
- **Feature extractor 128d đã đóng contract:** embedding deterministic từ bbox/position/velocity/heading/context để các phase sau có thể dùng ngay mà không chờ model học sâu.
- **Benchmark pipeline đã có nhánh chạy rõ mục đích:** `scripts/benchmark.py` tách workload perception/runtime, giữ cảnh báo khi dùng synthetic benchmark để tránh hiểu nhầm là số đo hardware truth.
- **Validation unit test đồng bộ hiện trạng code:** chạy `python -m pytest` tại `develop` cho kết quả `57 passed`, bao gồm detector/depth/lidar/tracker/imu/fusion/benchmark hooks.

## PHÂN HỆ III: PHASE 1.5 (Runtime Control Plane + Data Plane)

### 1. Control Plane (FastAPI) + Runtime State
- **Control API đã tách rõ khỏi data-plane:** thêm các endpoint `GET /health`, `GET /ready`, `GET /config`, `GET /metrics`, `POST /control/start`, `POST /control/stop` trong `src/api/app.py`.
- **Runtime controller có readiness gating theo tín hiệu thật:** `JetsonRuntimeController` chỉ báo `ready=true` khi đồng thời có engine, camera AstraS, stream LiDAR và IMU; có reason cụ thể cho từng trạng thái chờ (`waiting for TensorRT engine`, `waiting for lidar stream...`, `imu stream is stale`).
- **Config runtime chuẩn hoá env:** host/port Jetson-Pi, đường dẫn engine, camera devices, sensor freshness đều đi qua `RuntimeConfig` để deploy không phải hardcode rải rác.

### 2. Data Plane (Protobuf + ZMQ)
- **Protocol cảm biến đã chuẩn hóa wire format:** bổ sung `proto/sensors.proto` và codec `src/transport/messages.py` với envelope có `source_id`, `sequence`, `timestamp_us`, và payload loại `lidar_scan`/`imu_sample`/`pi_status`.
- **Protocol kết quả perception đã chuẩn hóa:** `proto/perception.proto` + `src/transport/results.py` đóng gói entities và runtime metrics thành `PerceptionResultEnvelope`.
- **Transport runtime đã tách đúng vai trò:** Jetson ingest bằng ZMQ `PULL` (`src/transport/zmq_sensor_ingest.py`) và publish kết quả bằng ZMQ `PUB` (`src/transport/zmq_result_publisher.py`); Pi bridge (`scripts/pi_sensor_bridge.py`) gửi JSONL sensor vào Jetson bằng ZMQ `PUSH`.

### 3. Deploy/Operations Support
- **Docker compose cho production/dev đã khớp control-runtime mới:** `control-api` chạy `uvicorn`, mount engine volume, map `/dev/video0` và `/dev/video1`, mở các cổng control + ingest + publish tương ứng.
- **Systemd service đã có entrypoint vận hành:** `deploy/jetson/adaptive-context-aware.service` dùng `docker compose up -d control-api` để boot runtime theo máy.
- **Rule runtime không fallback synthetic trong production:** tài liệu kiến trúc `docs/architecture/runtime-data-plane.md` đã khóa nguyên tắc không dùng dữ liệu giả khi thiếu camera/sensor thật.

## PHÂN HỆ IV: PHASE 1.6 (Stability + Compatibility)

### 1. Runtime Compatibility
- **Python baseline đồng bộ lên 3.11:** `pyproject.toml` và Dockerfiles đã align `requires-python >=3.11`, tránh mismatch runtime trước đó.
- **Dependency runtime được tách profile rõ:** core thêm `fastapi`, `uvicorn`, `protobuf`, `pyzmq`; optional `engine` giữ `ultralytics` để tránh kéo dependency nặng vào mọi context.
- **Test coverage bổ sung cho control/data plane:** thêm test cho API, runtime controller, sensor/result codecs, và ZMQ ingest để giảm rủi ro regression khi bước sang Phase 2.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Validation end-to-end trên Jetson với thiết bị thật
- **Mục tiêu:** xác nhận runtime control/data plane hoạt động đúng trong môi trường production (camera thật + network thật).
- **Hướng dẫn thực hiện chi tiết:**
  - Mở file `docker/docker-compose.yml` để kiểm tra lại `devices` và `CTX_JETSON_HOST/CTX_PI_HOST`.
  - Chạy `docker compose -f docker/docker-compose.yml up -d control-api` trên Jetson.
  - Kiểm tra control plane bằng:
    - `curl http://127.0.0.1:8080/health`
    - `curl http://127.0.0.1:8080/metrics`
    - `curl -X POST http://127.0.0.1:8080/control/start`
  - Đối chiếu reason trong `/metrics` để xác định thiếu engine/camera/sensor ở đâu.

### 2. Kiểm thử luồng Pi -> Jetson cho LiDAR/IMU ở data-plane
- **Mục tiêu:** xác nhận ingest thật làm runtime chuyển từ `waiting for ...` sang trạng thái sẵn sàng.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở file `scripts/pi_sensor_bridge.py` và kiểm tra mapping `kind` (`lidar_scan`, `imu_sample`, `pi_status`).
  - Trên Pi, stream JSONL sensor vào bridge:
    - `python scripts/pi_sensor_bridge.py --jetson-host 25.12.4.100 --port 5555 < sensor_stream.jsonl`
  - Trên Jetson, theo dõi:
    - `curl http://127.0.0.1:8080/metrics`
    - `docker logs <control-api-container>`
  - Xác nhận `messages_received` tăng và `decode_errors` giữ 0.

### 3. Validation TensorRT runtime cho detector trên Jetson
- **Mục tiêu:** xác nhận `TensorRTEngineRunner` load được `yolov8s.engine` thật và perception runtime chạy đầy đủ end-to-end.
- **Hướng dẫn thực hiện chi tiết:**
  - Build detector engine trên Jetson bằng `python scripts/bootstrap_engine.py --engine-dir models/engines --model-name yolov8s`.
  - Chạy một frame thật qua `PersonDetector(DetectorConfig(engine_path=Path("models/engines/yolov8s.engine")))`.
  - Giữ nguyên input contract `(480, 640, 3) -> [N, 6]` để không phá pipeline hiện có.
  - Chạy lại:
    - `python -m pytest`
    - `python scripts/benchmark.py --device perception --frames 1000`

