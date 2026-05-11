# Báo cáo Tiến trình: PHASE 2 (Adaptive Core — The Brain)

> Cập nhật: 2026-05-11
> Trạng thái: Hoàn thành

---

## PHÂN HỆ I: PHASE 2.0 (Custom Neural Network Core)

### 1. Complexity Estimator + Temporal Pathways
- **Complexity Estimator đã đóng contract gatekeeper:** thêm `models/complexity_estimator.py` với MLP `[36 → 64 → 32 → 4]`, `BatchNorm + ReLU`, input đúng schema `[crowd_density, motion_entropy, anomaly_score_prev, soh_budget, scene_embedding(32)]`, output logits cho `LOW/MED/HIGH/CRITICAL`. Test synthetic chứng minh model học được các case phân tầng complexity và giữ gradient flow ổn định.
- **GRU Pathway bắt short-term temporal context:** thêm `models/gru_pathway.py` dùng GRU 1-layer `input_size=128`, `hidden_size=64`, nhận `[B, T, 128]` và trả `[B, 64]`. Thiết kế này giữ đúng vai trò latency thấp cho LOW/MED mode và đã có test shape, wrong-shape rejection, gradient flow.
- **TCN Pathway bắt long-term temporal pattern:** thêm `models/tcn_pathway.py` với 3 residual causal dilated Conv1d blocks, dilation `[1, 2, 4]`, output `[B, 64]`. Test xác nhận causal behavior, `forward_sequence`, T=1 edge case và contract `[B, 128, T] → [B, 64]`.

### 2. Entity Interaction + Spatial Graph Pathways
- **Attention Pathway xử lý quan hệ liên thực thể:** thêm `models/attention_pathway.py` dùng 1-layer `MultiheadAttention`, 4 heads, pre/post LayerNorm, feedforward `[128 → 256 → 128]`, mean-pool theo entity axis để hỗ trợ N thay đổi. Test xác nhận attention weights shape `[B, 4, N, N]`, weights sum-to-1, single/multiple entity và gradient flow.
- **GNN Pathway xử lý topo không gian:** thêm `models/gnn_pathway.py` với custom 2-layer Graph Attention Network, layer 1 multi-head `128 → 128`, layer 2 `128 → 256`, adjacency mask, self-loops, và helper `build_adjacency(positions_xyz_m, threshold_m=2.0)`. Test cover graph construction theo khoảng cách, batched/unbatched adjacency, N=0, wrong-shape rejection và gradient flow.
- **Gated Fusion hợp nhất pathway outputs:** thêm `models/gated_fusion.py`, hỗ trợ input dạng mapping hoặc concatenated tensor, chuẩn hóa max input 512d từ `GRU(64)+TCN(64)+Attention(128)+GNN(256)`, mask inactive pathways, softmax gate sum-to-1, output unified reasoning vector `[B, 256]`. Đây là lớp khóa để router có thể bật 1, 2, 3 hoặc 4 pathways mà downstream Phase 3 vẫn nhận cùng một vector.

## PHÂN HỆ II: PHASE 2.1 (Runtime Integration + Adaptive Routing)

### 1. Complexity Runtime Wrapper + SoH Monitor
- **Estimator wrapper đã nối Phase 1 perception sang Phase 2 routing:** thêm `src/complexity/estimator.py` với `ComplexityEstimator`, `SceneComplexityMetrics`, `ComplexityEstimate`, `ComplexityLevel`. Wrapper dựng input `[1, 36]` từ `FusedEntity`, tính crowd density, motion entropy, anomaly score, SoH budget và scene embedding deterministic; runtime TensorRT `.engine` được lazy-load qua adapter để test không cần TensorRT thật trong CI.
- **SoH Monitor đã chuẩn hóa health budget:** thêm `src/complexity/soh_monitor.py`, parse `tegrastats` để lấy RAM used/total, `GPU@...C`, `GR3D_FREQ ...%`, rồi tính `soh_budget ∈ [0, 1]` theo bottleneck RAM pressure, GPU temp và GPU utilization. Test dùng provider giả để xác nhận budget giảm khi GPU nóng, RAM cao hoặc GPU gần bão hòa.

### 2. Rule-Based Adaptive Router
- **Adaptive Router đã implement routing table Phase 2:** thêm `src/router/adaptive_router.py`, map `LOW → GRU`, `MED → GRU+TCN`, `HIGH → GRU+TCN+Attention`, `CRITICAL → all pathways`. Decision trả `requested_level`, `effective_level`, `active_pathways`, latency budget, per-pathway target và `downgraded`.
- **SoH override đã hoạt động rõ ràng:** nếu `soh_budget < 0.5`, router hạ complexity xuống đúng 1 level, ví dụ `CRITICAL → HIGH`, để tránh bật GNN khi GPU/RAM đang nguy hiểm. Test xác nhận LOW không bị hạ dưới LOW, HIGH hạ về MED, và invalid level bị reject.

### 3. Lazy Inference Wrappers
- **Pathway wrappers tách TensorRT runtime khỏi model definition:** thêm `src/reasoning/runtime.py` và các wrapper `gru_pathway.py`, `tcn_pathway.py`, `attention_pathway.py`, `gnn_pathway.py`. Mỗi wrapper validate shape đầu vào, chỉ tạo `TensorRTEngineRuntime` khi `infer()` được gọi, nên RAM chỉ tăng khi router thật sự activate pathway.
- **Fusion inference wrapper khớp với runtime contract:** thêm `src/reasoning/fusion.py`, nhận mapping hoặc concatenated active outputs, pad về `[B, 512]`, tạo `active_mask [B, 4]`, lazy-load `models/engines/gated_fusion.engine`, và đảm bảo output `[B, 256]` bất kể số pathway active.
- **Brain pipeline orchestration đã có integration surface:** thêm `src/reasoning/brain_pipeline.py` để chạy flow `Estimator → Router → Active Pathways → Fusion`, trả `BrainPipelineResult` gồm estimate, routing, pathway outputs, unified output và latency. Integration test dùng fake runtimes để kiểm đủ LOW/MED/HIGH/CRITICAL và SoH downgrade mà không cần TensorRT thật trong CI.

## PHÂN HỆ III: PHASE 2.2 (Training + Export Pipelines)

### 1. Complexity Estimator Training
- **Estimator training pipeline đã có synthetic labeled scenes:** thêm `pipelines/train_estimator.py`, sinh dataset 4 mức complexity với bands riêng cho crowd density, motion entropy, anomaly score, SoH và scene embedding. Training dùng `CrossEntropyLoss`, `AdamW`, lưu checkpoint `models/checkpoints/complexity_estimator.pt`; TensorRT `.engine` được build bằng bước riêng trên Jetson.
- **Validation accuracy được test ở mức pipeline:** unit test chạy train ngắn không phụ thuộc package TensorRT trong CI, nhưng vẫn xác nhận validation accuracy ≥90%, checkpoint tồn tại và dataset contract `[N, 36]`/labels `{0,1,2,3}` đúng.

### 2. Joint Reasoning Training
- **JointReasoningNet gom đủ bốn pathway và fusion:** thêm `pipelines/train_reasoning.py`, forward qua GRU, TCN, Attention, GNN, rồi GatedFusion; sau fusion có heads cho activity classification, direction classification, anomaly logits và reconstruction.
- **Multi-task loss đã cụ thể hóa mục tiêu Phase 2:** loss gồm activity CE, direction CE, anomaly BCE-with-logits và auxiliary reconstruction MSE hệ số 0.1. Synthetic dataset encode tín hiệu activity/direction/anomaly vào sequence/entity features để train loop đi qua toàn bộ graph thật nhưng vẫn chạy nhanh trong CI.
- **Checkpoint reasoning brain đã có đường chuẩn:** `train_reasoning()` lưu `models/checkpoints/reasoning_brain.pt`, test xác nhận output contract `fused [B,256]`, `activity [B,4]`, `direction [B,9]`, `anomaly [B]`, `reconstruction [B,128]`, loss finite và checkpoint tồn tại.

### 3. TensorRT Engine Build Script
- **Brain TensorRT builder cover đủ 6 model Phase 2:** thêm `scripts/build_brain_engines.py`, build estimator, GRU, TCN, Attention, GNN và Fusion ra `models/engines/*.engine`. Mỗi model có `BrainEngineSpec` riêng với input shapes, expected output shape và artifact metadata.
- **Fusion build adapter giữ đúng runtime input:** `FusionEngineWrapper` build fusion theo input `[pathway_outputs, active_mask]`, khớp với `GatedFusionInference` thay vì build API Python mapping vốn không phù hợp TensorRT input contract.
- **CI không cần TensorRT thật:** unit test dùng fake `EngineBuilder` để xác nhận đủ 6 engine artifact và metadata được ghi; shape check PyTorch vẫn chạy để bắt lỗi contract trước khi build trên Jetson.

## PHÂN HỆ IV: PHASE 2.3 (Testing, Benchmark, Git Integration)

### 1. Test Coverage
- **Unit test từng network đã bổ sung đầy đủ:** thêm các test riêng cho Complexity Estimator, GRU, TCN, Attention, GNN, Gated Fusion, cộng thêm `tests/unit/test_brain_networks.py` để smoke toàn bộ brain networks, gradient flow, T=1, single entity và empty graph.
- **Runtime wrapper tests xác nhận lazy loading:** `tests/unit/test_reasoning_pathway_wrappers.py` và `tests/unit/test_fusion_inference_wrapper.py` dùng fake runtime factory để chứng minh wrapper chưa load khi chưa active, reuse runtime sau lần infer đầu, reject wrong shapes và reject output mismatch.
- **Brain integration test xác nhận routing end-to-end:** `tests/unit/test_brain_pipeline_integration.py` chạy đủ 4 complexity levels, kiểm pathway nào được gọi, pathway inactive không bị gọi, fusion nhận đúng active set, output unified `[B, 256]`, và SoH downgrade loại GNN khỏi CRITICAL khi budget thấp.

### 2. Benchmark Baselines
- **TCN đã được thêm vào baseline CI:** cập nhật `scripts/benchmark.py`, `tests/benchmark/baselines/latency_baseline.json`, `baseline_meta.json`, và thêm `tests/benchmark/baselines/output_reference/tcn_ref.npy`.
- **Benchmark compare đã tự-consistent:** `python scripts/benchmark.py --device ci --compare-baseline` pass với metrics synthetic gồm `gru_pathway`, `tcn_pathway`, `attention_pathway`, `gnn_pathway`, `complexity_estimator`, `rl_policy`.

### 3. Git Flow + Merge State
- **Mỗi task Phase 2 được implement trên branch riêng:** các task T2.1-T2.17 được commit theo feature branch riêng, merge về `phase/2-adaptive-core`, sau đó phase branch được merge vào `develop` qua PR #7.
- **Phase branches đã dọn sạch:** sau khi merge Phase 1 và Phase 2 vào `develop`, các branch local/remote `phase/1-perception` và `phase/2-adaptive-core` đã bị xóa; hiện `develop` sạch và sync với `origin/develop`.
- **Verification cuối trên develop:** chạy `python -m ruff check .` pass, `python -m pytest` pass với `132 passed`, và `python scripts\benchmark.py --device ci --compare-baseline` pass.

---

## Các bước tiếp theo cần làm (Next Steps)

### 1. Build TensorRT `.engine` thật cho toàn bộ Phase 2 Brain
- **Mục tiêu:** tạo đủ 6 file TensorRT engine thật dưới `models/engines/` để runtime wrappers có artifact inference.
- **Hướng dẫn thực hiện chi tiết:**
  - Thực hiện trên Jetson Nano đúng JetPack/TensorRT runtime sẽ deploy; `.engine` là artifact phụ thuộc phần cứng.
  - Cài/kiểm tra runtime binding trong image Jetson: `python -c "import tensorrt, torch_tensorrt"`.
  - Chạy builder từ repo root:
    - `python scripts/build_brain_engines.py --output-dir models/engines`
  - Kiểm tra file output:
    - `models/engines/estimator.engine`
    - `models/engines/gru_pathway.engine`
    - `models/engines/tcn_pathway.engine`
    - `models/engines/attention_pathway.engine`
    - `models/engines/gnn_pathway.engine`
    - `models/engines/gated_fusion.engine`
  - Nếu builder fail ở Fusion, mở `scripts/build_brain_engines.py`, kiểm tra `FusionEngineWrapper.forward()` và input `active_mask [B,4]`.
  - Chạy lại test liên quan:
    - `python -m pytest tests/unit/test_build_brain_engines.py`
    - `python -m pytest tests/unit/test_reasoning_pathway_wrappers.py tests/unit/test_fusion_inference_wrapper.py`

### 2. Chạy training pipeline để tạo checkpoint thật trước khi build production engine
- **Mục tiêu:** thay checkpoint/synthetic smoke bằng weights đã train có ý nghĩa hơn cho estimator và reasoning brain.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `pipelines/train_estimator.py`, điều chỉnh `EstimatorTrainingConfig.samples_per_level`, `epochs`, `learning_rate` nếu muốn train lâu hơn.
  - Chạy estimator:
    - `python -m pipelines.train_estimator --samples-per-level 512 --epochs 300 --checkpoint-path models/checkpoints/complexity_estimator.pt`
  - Mở `pipelines/train_reasoning.py`, kiểm tra `ReasoningTrainingConfig.samples`, `sequence_length`, `entity_count`.
  - Chạy joint reasoning:
    - `python -m pipelines.train_reasoning --samples 1024 --epochs 50 --checkpoint-path models/checkpoints/reasoning_brain.pt`
  - Sau training, bổ sung loader checkpoint vào `scripts/build_brain_engines.py` nếu muốn build từ weights đã train thay vì model init ngẫu nhiên.

### 3. Kết nối Phase 2 Brain vào runtime perception loop
- **Mục tiêu:** sau khi Phase 1 tạo `FusedEntity` và entity features, runtime gọi được Phase 2 để ra `unified_output [B,256]`.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `src/perception/pipeline.py`; hiện pipeline mới extract `_features` nhưng chưa chuyển sang Phase 2.
  - Tạo adapter build:
    - `sequence_features [B,T,128]` cho GRU/TCN từ history entity embeddings.
    - `entity_features [B,N,128]` cho Attention/GNN từ current frame.
    - `adjacency [B,N,N]` bằng `GraphAttentionPathway.build_adjacency()` hoặc numpy equivalent.
  - Instantiate `AdaptiveBrainPipeline` từ `src/reasoning/brain_pipeline.py` với `ComplexityEstimator` và runtime wrappers thật.
  - Truyền `soh_budget` từ `SoHMonitor.sample().soh_budget`.
  - Bổ sung integration test mới ở `tests/unit/test_runtime_controller.py` hoặc file mới `tests/unit/test_perception_brain_bridge.py`.
  - Chạy:
    - `python -m pytest tests/unit/test_brain_pipeline_integration.py`
    - `python -m pytest`

### 4. Chuẩn bị Phase 3 Behavior & Decision trên output `[B,256]`
- **Mục tiêu:** dùng unified reasoning vector từ Phase 2 làm input chính cho intent/anomaly/navigation behavior.
- **Hướng dẫn thực hiện chi tiết:**
  - Mở `docs/plan/PLAN-adaptive-context-aware.md` và tạo/đọc plan Phase 3 tương ứng.
  - Định nghĩa model/runtime contract Phase 3 dự kiến:
    - intent direction head: `[B,256] → direction classes`
    - activity head: `[B,256] → activity classes`
    - anomaly head: `[B,256] → probability`
    - navigation hint: `[B,256] + robot state → action/hint`
  - Tạo file model đầu tiên dưới `models/` hoặc wrapper dưới `src/behavior/` tùy kiến trúc Phase 3.
  - Bổ sung test contract tương tự Phase 2 trước khi train:
    - `python -m pytest tests/unit/test_model_contracts.py`

### 5. Chạy benchmark thật trên Jetson sau khi có TensorRT `.engine`
- **Mục tiêu:** thay synthetic CI numbers bằng số đo hardware thật cho LOW/MED/HIGH/CRITICAL.
- **Hướng dẫn thực hiện chi tiết:**
  - Trên Jetson, đảm bảo `models/engines/*.engine` đã có trong volume/model directory.
  - Chạy:
    - `python scripts/benchmark.py --device jetson --frames 1000`
  - Nếu cần baseline mới cho CI/hardware class:
    - `python scripts/update_ci_baselines.py --source jetson-orin-nano-8gb --output tests/benchmark/baselines`
  - Mở `tests/benchmark/baselines/latency_baseline.json`, cập nhật riêng budgets cho `complexity_estimator`, `gru_pathway`, `tcn_pathway`, `attention_pathway`, `gnn_pathway`, và `gated_fusion` nếu thêm vào benchmark metrics.
  - Chạy lại:
    - `python scripts/benchmark.py --device ci --compare-baseline`
    - `python -m pytest tests/unit/test_benchmark_script.py`
