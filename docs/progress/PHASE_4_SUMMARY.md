# Báo cáo Tiến trình: PHASE 4 (RL Policy for Adaptive Router)

> Cập nhật: 2026-05-18
> Trạng thái: Hoàn thành phần cốt lõi, còn một số hạng mục nâng cao chưa đóng

---

## PHÂN HỆ I: PHASE 4.1 (RL Environment + State/Reward Contract)

### 1. Simulation environment
- **Adaptive routing environment đã được implement:** `src/router/rl_policy.py` có `AdaptiveRoutingEnv` với `reset()` và `step()` trả về tuple kiểu Gym/Gymnasium gồm `state`, `reward`, `terminated`, `truncated`, `info`.
- **Fixture scenarios đã có để bootstrap training/evaluation:** `tests/fixtures/rl_scenarios/` cung cấp các scenario JSON để load qua `AdaptiveRoutingEnv.from_fixture_dir(...)`.

### 2. State / action / reward
- **State contract bám sát plan Phase 4:** `RLRouterState` đóng gói `crowd_density`, `motion_entropy`, `anomaly_probability`, `soh_budget`, `scene_embedding`, `prev_action`, `time_since_critical`, `current_fps` và encode thành vector `[1,39]`.
- **Action space đã được chuẩn hóa cho router:** `RoutingAction = {FAST, BALANCED, ACCURATE, EMERGENCY}` tương ứng với các mức pathway từ nhẹ đến đầy đủ.
- **Reward function đã có accuracy-latency-energy tradeoff:** `evaluate_action(...)` kết hợp `accuracy`, `latency_penalty`, `energy_penalty` và `anomaly_bonus` thay vì chỉ dùng một score đơn.

## PHÂN HỆ II: PHASE 4.2 - 4.4 (Training, Engine Path, Router Integration)

### 1. Training pipeline
- **Pipeline train RL router đã có:** `pipelines/train_router_rl.py` sinh dataset từ scenario fixtures, train `RLPolicyNet`, lưu checkpoint và báo `validation_accuracy`, `reward_before`, `reward_after`.
- **Có hỗ trợ Stable-Baselines3 theo hướng opportunistic:** backend `ppo` được dùng nếu môi trường có `gymnasium` và `stable_baselines3`; nếu không có thì fallback sang backend supervised để CI/dev machine vẫn chạy được.
- **Test cho training contract đã pass:** `tests/unit/test_train_router_rl_pipeline.py` xác nhận dataset shape đúng, checkpoint được lưu và reward sau train không tệ hơn baseline.

### 2. Runtime policy + engine path
- **Runtime RL policy đã có abstraction rõ ràng:** `RLPolicy` nhận state `[B,39]`, chạy runtime và decode logits thành `RoutingAction`.
- **Đường build TensorRT engine đã được nối vào pipeline build engine:** `scripts/build_brain_engines.py` đã thêm spec `rl_policy.engine`, nghĩa là Phase 4 đã có “engine path” trong build graph.
- **Phần benchmark `<1ms` chưa thấy bằng chứng chốt trong repo:** hiện chưa có report hoặc test benchmark riêng xác nhận inference của RL policy dưới 1ms trên Jetson.

### 3. Integration vào adaptive router
- **Adaptive router đã hỗ trợ dùng RL policy thật:** `src/router/adaptive_router.py` nhận `rl_policy`, map action RL sang `ComplexityLevel`, rồi chọn pathway tương ứng.
- **Fallback khi RL lỗi đã được implement đúng yêu cầu plan:** nếu `rl_policy.decide(...)` ném exception thì router quay về chiến lược rule-based và đánh dấu `fallback_used=True`.
- **Test integration/fallback đã pass:** `tests/unit/test_adaptive_router.py` cover cả nhánh dùng RL thành công lẫn nhánh fallback.

## PHÂN HỆ III: Đánh giá mức hoàn thành so với plan

### 1. Các mục đã hoàn thành hoặc gần hoàn thành
- **T4.1 đã hoàn thành:** env, state, action, reward và contract test đều đã có.
- **T4.2 đã hoàn thành theo mức bootstrap khả dụng:** training script và test coverage đã có; tuy nhiên repo mới chứng minh “train được và cải thiện reward fixture”, chưa có training report dài hạn với reward curve đầy đủ.
- **T4.4 đã hoàn thành:** RL policy đã tích hợp vào `adaptive_router.py` và có fallback rule-based.

### 2. Các mục còn mở
- **T4.3 mới hoàn thành ở mức pipeline/build path:** đã có model + engine spec, nhưng chưa thấy artifact benchmark để xác nhận mục tiêu `<1ms` trên Jetson.
- **T4.5 chưa thấy implement:** chưa có cơ chế online adaptation/fine-tune trên Jetson trong repo hiện tại.
- **“RL outperforms rule-based” mới được kiểm chứng ở mức fixture reward:** pipeline có so `reward_before` và `reward_after`, nhưng chưa có báo cáo evaluation rộng hơn trên test scenarios hoặc hardware run thật.

---

## Kết luận ngắn

- **Có thể xem Phase 4 đã được implement ở phần cốt lõi.**
- **Nếu cần chốt “Done When” theo plan gốc thì Phase 4 hiện nên được gọi là `substantially implemented / partially closed`, chưa phải đóng hoàn toàn.**

## Bằng chứng verify đã chạy

- **Local test pass:** `python -m pytest tests/unit/test_rl_policy.py tests/unit/test_adaptive_router.py tests/unit/test_train_router_rl_pipeline.py -q`
- **Kết quả:** `13 passed`
