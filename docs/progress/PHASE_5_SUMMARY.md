# Báo cáo Tiến trình: PHASE 5 (Safety & Monitoring)

> Cập nhật: 2026-05-12
> Trạng thái: Hoàn thành phần software core

---

## PHÂN HỆ I: PHASE 5.0 (Safety Core)

### 1. Safety FSM
- **State machine đã được nâng lên mức reject transition sai rõ ràng:** `src/safety/state_machine.py` hiện có `InvalidTransitionError`, giữ logic chặn `ESTOP -> NORMAL`, `ESTOP -> DEGRADED`, `RECOVERY -> DEGRADED`, `NORMAL -> RECOVERY`.
- **Unit tests đã khóa behavior state machine:** `tests/unit/test_safety_fsm.py` kiểm cả transition hợp lệ và transition bị cấm.

### 2. ESTOP + Recovery
- **ESTOP command đã được chuẩn hóa thành object rõ nghĩa:** `src/safety/estop.py` sinh `EStopCommand` với motor disable + zero velocity, phản ánh nguyên tắc “Jetson advisory, RPi executes”.
- **Recovery self-check đã có skeleton chạy được:** `src/safety/recovery.py` cho phép self-test suite trả `RecoveryCheckResult`, và `tests/unit/test_estop_recovery.py` cover pass/fail path.

### 3. Graceful Degradation
- **Degradation logic đã đóng thành decision rõ ràng:** `src/safety/graceful_degrade.py` map các case `gpu_oom`, `camera_failed`, `gpu_temp_c > 80` sang mode degrade cụ thể.
- **Mode degrade đã phản ánh đúng spec tinh gọn:** có `gru_only`, `lidar_only`, `thermal_throttle` thay vì để runtime tự đoán fallback.

## PHÂN HỆ II: PHASE 5.1 (Watchdog + Metrics + Logger)

### 1. Watchdog
- **Process watchdog đã được thêm độc lập với heartbeat watchdog trong comm layer:** `src/safety/watchdog.py` xử lý logic restart timeout ở mức process/container health.
- **Integration heartbeat->ESTOP vẫn tiếp tục dùng `src/comm/health_monitor.py`:** `tests/integration/test_heartbeat_estop.py` đã khóa mốc timeout ~2s đúng spec.

### 2. Metrics
- **Safety metrics payload đã có format thống nhất:** `src/safety/metrics.py` đóng gói FPS, latency, GPU/CPU/RAM và phân phối pathway selection.
- **Payload JSON-ready:** downstream API hoặc Prometheus adapter có thể bám vào `as_json_payload()` thay vì map từng field thủ công.

### 3. Logger
- **Logger builder đã có env-sensitive levels:** `src/safety/logger.py` map `dev=DEBUG`, `test=INFO`, `prod=WARNING`.
- **Test helper đã cover config level:** `tests/unit/test_safety_helpers.py` giữ guard để logging layer không drift ngược khỏi spec.

## PHÂN HỆ III: PHASE 5.2 (Coverage + Integration)

### 1. Test layer
- **Safety helper tests đã cover logic cốt lõi:** degrade decisions, metrics payload naming, logger levels, watchdog restart threshold.
- **Heartbeat ESTOP integration mock đã có:** đủ để software layer tiến tiếp mà chưa cần phần cứng thật ngay.

### 2. Những phần chưa đóng phần cứng
- **T5.6 systemd auto-start on boot** chưa được triển khai/verify trong lượt này.
- **Motor-stop ownership end-to-end** mới được đảm bảo ở mức code contract, chưa phải test vật lý với RPi/hardware relay.
- **No auto-recovery / manual reset** đã được biểu diễn ở FSM level, nhưng chưa nối full runtime orchestration ngoài test mock.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Nối safety core vào runtime thật
- **Mục tiêu:** để `runtime controller` không chỉ báo trạng thái mà còn thật sự drive `SafetyStateMachine`, degrade decisions, và ESTOP orchestration.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `src/runtime/controller.py`.
  - Inject `SafetyStateMachine`, `decide_degradation()`, và `build_estop_command()` vào control flow runtime.

### 2. Hoàn tất T5.6 systemd / deployment
- **Mục tiêu:** hoàn chỉnh phần auto-start safety/deploy trên Jetson.
- **Hướng dẫn thực hiện chi tiết:**
  - Bổ sung hoặc cập nhật service file dưới `deploy/`.
  - Verify lại path docker compose, restart policy, watchdog boot order.

### 3. Mở rộng integration tests
- **Mục tiêu:** chuyển từ helper-level sang runtime-level integration.
- **Hướng dẫn thực hiện chi tiết:**
  - Thêm test `heartbeat loss -> ESTOP -> manual recovery -> NORMAL` với nhiều thành phần runtime hơn.
  - Chạy:
    - `python -m pytest tests/unit/test_safety_fsm.py tests/unit/test_estop_recovery.py tests/unit/test_safety_helpers.py tests/integration/test_heartbeat_estop.py`

