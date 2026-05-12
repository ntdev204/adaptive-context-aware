# Báo cáo Tiến trình: PHASE 0.5 (Data Foundation)

> Cập nhật: 2026-05-12
> Trạng thái: Hoàn thành

---

## PHÂN HỆ I: PHASE 0.5.1 (Annotation Schema + Enums)

### 1. Schema và Enum chuẩn hóa
- **Annotation schema đã chốt theo spec:** `config/schemas/annotation_schema.json` bám đúng `docs/specs/data-schema.md` cho `frame_id`, `persons[]`, `scene.context`, `crowd_density`, `motion_entropy`, `anomaly_flag`.
- **Enum dùng chung đã ổn định:** `src/utils/enums.py` là nguồn chuẩn cho `Activity`, `SceneContext`, `IntentDirection`, đồng thời các phase downstream có thể import trực tiếp thay vì tự định nghĩa lại.
- **Validation test không còn chỉ dựa trên mẫu hardcode:** `tests/unit/test_data_schema.py` đã validate trực tiếp các fixture annotation committed và có negative case để bắt enum sai.

## PHÂN HỆ II: PHASE 0.5.2 (HDF5 Record / Replay)

### 1. HDF5 contract đã usable thật
- **Recorder đã ghi đủ nhóm dữ liệu theo spec:** `src/utils/hdf5_recorder.py` ghi `/metadata`, `/rgb_frames`, `/depth_frames`, `/lidar_scans`, `/imu`, và `/annotations` đúng cấu trúc HDF5 đã mô tả.
- **Reader không chỉ `read()` mà còn `iter_batches()`:** `src/utils/hdf5_reader.py` đã có replay iterator theo frame để downstream pipeline dùng như data stream thay vì phải load toàn bộ file vào RAM.
- **Roundtrip test đủ để tin cậy contract:** `tests/unit/test_hdf5_io.py` cover record -> read -> replay alignment giữa RGB/depth/LiDAR/IMU/annotations.

## PHÂN HỆ III: PHASE 0.5.3 (Bootstrap Fixtures)

### 1. Bộ dữ liệu mồi đã hoàn thiện
- **Committed fixtures đã được bổ đủ mức bootstrap:** `tests/fixtures/annotations/` có 5 frame mẫu, `anomaly_synthetic/` có 5 case, `rl_scenarios/` có 5 scenario, `images/` có 5 ảnh nhỏ, và `sample_recording.h5` tồn tại.
- **Không còn phụ thuộc COCO/MOT17 download ngoài:** `scripts/download_fixtures.py` đã chuyển sang extract subset cục bộ từ `data/fine_tuning/cctv_person` và `data/fine_tuning/mot20`, sinh `tests/fixtures/cctv_person_subset/` và `tests/fixtures/mot20_subset/`.
- **Generator fixture đã bao phủ cả HDF5 và ảnh nhỏ:** `scripts/generate_synthetic_fixtures.py` sinh lại `sample_recording.h5` và 5 image fixtures trong cùng một bước.
- **`.gitignore` và docs đã đổi theo chiến lược local subset:** repo không còn định hướng “download COCO/MOT17” cho bootstrap nữa.

## PHÂN HỆ IV: PHASE 0.5.4 (Model Contract Tests)

### 1. Contract layer đã làm nền cho các phase sau
- **Model I/O contract tests đã tồn tại và được dùng xuyên phase:** `tests/unit/test_model_contracts.py` hiện đang kiểm shape/dtype cho detector, estimator, pathways, intent/anomaly heads và RL policy.
- **Contract detector đã sync với `YOLO11-s`:** sau khi detector đổi sang `yolo11s`, contract shape `[1, 3, 480, 640] -> [N, 6]` vẫn được giữ nguyên để không phá downstream.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Dùng bootstrap subset để sanity-check perception training
- **Mục tiêu:** tận dụng ngay `cctv_person_subset` và `mot20_subset` làm smoke test cho pipeline train/eval cục bộ.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `tests/fixtures/cctv_person_subset/` và `tests/fixtures/mot20_subset/`.
  - Kiểm tra script trích subset: `scripts/download_fixtures.py`.
  - Nếu cần regenerate:
    - `python scripts/download_fixtures.py`
    - `python scripts/generate_synthetic_fixtures.py`

### 2. Chuẩn hóa derived dataset trên cùng contract HDF5
- **Mục tiêu:** để reasoning layer không phải phát minh format mới ngoài HDF5 contract đã có.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `src/utils/hdf5_recorder.py` và `src/utils/hdf5_reader.py`.
  - Thiết kế thêm group `/derived/...` cho sequence/graph samples nếu muốn reasoning layer đọc chung một container.

### 3. Giữ fixture tests trong CI như smoke barrier
- **Mục tiêu:** tránh các phase sau vô tình làm hỏng bootstrap data layer.
- **Hướng dẫn thực hiện chi tiết:**
  - Chạy:
    - `python -m pytest tests/unit/test_fixtures.py tests/unit/test_data_schema.py tests/unit/test_hdf5_io.py`

