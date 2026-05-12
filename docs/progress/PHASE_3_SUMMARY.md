# Báo cáo Tiến trình: PHASE 3 (Behavior & Decision)

> Cập nhật: 2026-05-12
> Trạng thái: Hoàn thành một phần có chủ đích

---

## PHÂN HỆ I: PHASE 3.0 (Intent + Anomaly + Navigation)

### 1. Intent Prediction
- **Intent predictor đã có 3 head đúng contract:** thêm `models/intent_predictor.py` và `src/decision/intent_predictor.py`, nhận `unified_output [B,256]` và sinh `direction_logits`, `activity_logits`, `trajectory_offsets`.
- **Enums và prediction object đã đủ cho downstream dùng trực tiếp:** `IntentPrediction`, `IntentDirection`, `ActivityClass` cho phép decision layer làm việc mà không cần parse tensor thô ở nhiều nơi.

### 2. Anomaly Detection
- **Anomaly detector dùng nhiều tín hiệu thay vì một score đơn lẻ:** `src/decision/anomaly_detector.py` kết hợp statistical score, learned reconstruction proxy, temporal change và activity prior.
- **Synthetic fixture evaluation đã có:** logic detector được test với anomaly fixtures committed để giữ ngưỡng recall bootstrap khả dụng trong CI.

### 3. Navigation Command
- **Nav commander đã có logic tránh va chạm theo intent/anomaly:** `src/decision/nav_commander.py` kết hợp goal robot, predicted entity motion và anomaly score để sinh `velocity_xy_mps`, `omega_radps`, `mode`.
- **Chế độ `PROCEED / AVOID / HOLD` đã được chuẩn hóa:** giúp downstream runtime/safety layer đọc command ở mức semantic thay vì chỉ nhìn vector vận tốc.

## PHÂN HỆ II: PHASE 3.1 (Behavior Pipeline Integration)

### 1. Full decision pipeline
- **BehaviorDecisionPipeline đã nối thẳng từ output Phase 2:** `src/decision/behavior_pipeline.py` chạy `IntentPredictor -> AnomalyDetector -> NavigationCommander` trên `BrainPipelineResult`.
- **Integration tests đã cover end-to-end behavior output:** `tests/unit/test_behavior_pipeline.py` xác nhận intent/anomaly/nav command được sinh ra đồng bộ từ `unified_output`.

### 2. Contract và test coverage
- **Model contract test đã được mở rộng cho Phase 3:** `tests/unit/test_model_contracts.py` giữ shape contract cho intent/anomaly outputs.
- **Unit test riêng cho từng module decision đã có đủ:** `test_intent_predictor.py`, `test_anomaly_detector.py`, `test_nav_commander.py`, `test_behavior_pipeline.py`.

## PHÂN HỆ III: PHASE 3.5 (Scope Deliberately Deferred)

### 1. Scene context classifier
- **T3.4 được giữ lại như deferred work:** phần `scene context classifier` chưa được implement trong lượt này vì verify của plan đòi test-image/context path riêng, dễ kéo theo perception/domain labeling chưa chốt.
- **Schema đã giữ context set đầy đủ:** `RESTAURANT` đã được restore lại trong enums/schema/spec để future scene classifier không bị mất nhánh context.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Nếu muốn đóng nốt T3.4
- **Mục tiêu:** thêm scene context classifier đúng nghĩa cho `CORRIDOR / LOBBY / GATE_AREA / RESTAURANT / OPEN_SPACE`.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `docs/plan/PLAN-phase-3-behavior.md`.
  - Thiết kế riêng model/context head hoặc rule-based bootstrap từ perception classes.
  - Bổ sung test image fixtures cho từng context trước khi train model thật.

### 2. Nối Phase 3 vào runtime controller
- **Mục tiêu:** sau Phase 2 brain output, runtime thật gọi luôn behavior pipeline.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `src/perception/pipeline.py` và `src/runtime/controller.py`.
  - Chèn `BehaviorDecisionPipeline.run(...)` sau khi có `BrainPipelineResult`.

