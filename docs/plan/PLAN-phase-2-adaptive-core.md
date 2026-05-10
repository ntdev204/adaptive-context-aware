# Phase 2: Adaptive Core — The Brain

> Complexity Estimator + Adaptive Router + GRU/TCN/Attention/GNN Pathways + Fusion
>
> **Nguyên tắc cốt lõi:** TẤT CẢ mạng nơ-ron trong phase này đều được TỰ THIẾT KẾ từ đầu bằng PyTorch, tối ưu cho Jetson Orin Nano 8GB. Không dùng model có sẵn — chúng ta xây dựng bộ não riêng.

---

## Goal

Implement hệ thống AI "nghĩ tỷ lệ thuận với độ phức tạp" — lightweight khi đơn giản, deep khi phức tạp. Bốn mạng chuyên gia (GRU, TCN, Attention, GNN) hoạt động như một bộ não hoàn chỉnh, được điều phối bởi Adaptive Router và kết hợp tại Gated Fusion Layer.

---

## Brain Architecture Overview

```
                    ┌──────────────────┐
                    │  Entity Features │  ← from Phase 1 Perception (128-dim per entity)
                    │  (B, N, 128)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    Complexity     │
                    │    Estimator      │  ← custom MLP (tiny, <1ms)
                    │    (36-dim → 4)   │
                    └────────┬─────────┘
                             │
              complexity_level + soh_budget
                             │
                    ┌────────▼─────────┐
                    │  Adaptive Router  │  ← rule-based (Phase 2) → RL (Phase 4)
                    │                   │
                    │  Decides WHICH    │
                    │  pathways to      │
                    │  activate         │
                    └───┬──┬──┬──┬──────┘
                        │  │  │  │
          ┌─────────────┘  │  │  └─────────────┐
          │                │  │                │
    ┌─────▼─────┐  ┌──────▼──▼──┐  ┌──────────▼──────────┐
    │    GRU    │  │    TCN     │  │     Attention       │
    │  Pathway  │  │  Pathway   │  │     Pathway         │
    │           │  │            │  │                     │
    │ 1-layer   │  │ 3-block    │  │ Multi-head (4h)     │
    │ hidden=64 │  │ dilated    │  │ dim=128             │
    │           │  │ causal CNN │  │                     │
    │ ~1ms      │  │ ~3ms       │  │ ~5ms                │
    └─────┬─────┘  └──────┬────┘  └──────────┬──────────┘
          │               │                  │
          │        ┌──────▼──────┐           │
          │        │    GNN      │           │
          │        │  Pathway    │           │
          │        │             │           │
          │        │ 2-layer GAT │           │
          │        │ spatial     │           │
          │        │ graph       │           │
          │        │ ~10ms       │           │
          │        └──────┬──────┘           │
          │               │                  │
    ┌─────▼───────────────▼──────────────────▼─────┐
    │              Gated Fusion Layer               │
    │                                               │
    │  Learns weighted combination of all active    │
    │  pathway outputs → unified reasoning vector   │
    └───────────────────┬───────────────────────────┘
                        │
                  unified_output
                   (256-dim)
                        │
                        ▼
              Phase 3: Behavior & Decision
              (Intent, Anomaly, Navigation)
```

### Pathway Roles — Tại Sao Cần Cả 4?

| Pathway | Vai Trò Chuyên Biệt | Ví Dụ Thực Tế |
|---------|---------------------|----------------|
| **GRU** | Short-term temporal: thay đổi tức thì (~0.5-2s) | Người đột ngột dừng lại, đổi hướng |
| **TCN** | Long-term temporal: pattern kéo dài (3-10s+) | Người đi vòng vòng (loitering), tụ tập dần |
| **Attention** | Inter-entity relationships: ai tương tác với ai | 2 người tiến gần nhau, nhóm tụ họp |
| **GNN** | Spatial structure: topo đồ thị không gian | Tắc nghẽn cửa ra, luồng người di chuyển, cụm đám đông |

> **Key insight:** GRU bắt thay đổi nhanh, TCN bắt xu hướng chậm, Attention bắt tương tác giữa các thực thể, GNN bắt cấu trúc không gian. Cả 4 bổ sung cho nhau — **không thừa, không thiếu.**

---

## Routing Strategy

| Complexity Level | Active Pathways | Total Latency Budget | Use Case |
|------------------|----------------|---------------------|----------|
| **LOW** (0) | GRU only | <5ms | Hành lang vắng, 1-2 người đi thẳng |
| **MED** (1) | GRU + TCN | <10ms | Vài người, có pattern cần theo dõi |
| **HIGH** (2) | GRU + TCN + Attention | <20ms | Đông đúc, nhiều tương tác |
| **CRITICAL** (3) | ALL (GRU + TCN + Attention + GNN) | <35ms | Gate area, anomaly, fighting, đám đông lớn |

> **SoH Override:** Nếu GPU quá nóng hoặc RAM sắp cạn → Router tự động hạ cấp (ví dụ: HIGH → MED) bất kể complexity level.

---

## Tasks

### Phần A: Custom Neural Network Design (Tự Thiết Kế Mạng)

- [ ] **T2.1**: Design & implement `models/complexity_estimator.py` — Custom MLP
  - **Architecture:** 3-layer MLP `[36 → 64 → 32 → 4]` với ReLU + BatchNorm
  - **Input (36-dim):** `[crowd_density, motion_entropy, anomaly_score_prev, soh_budget, scene_embedding(32)]`
  - **Output (4-dim):** logits cho `{LOW=0, MED=1, HIGH=2, CRITICAL=3}`
  - **Parameters:** ~3.5K params (~14KB)
  - **Design rationale:** Tiny MLP vì đây là gatekeeper — chạy MỌI frame, phải cực nhanh
  → Verify: Forward pass <1ms trên CPU, correct classification trên synthetic test cases

- [ ] **T2.2**: Design & implement `models/gru_pathway.py` — Custom GRU Network
  - **Architecture:** 1-layer GRU, hidden_size=64, input_size=128
  - **Input:** `[B, T, 128]` — entity features qua T timesteps (T=8 default)
  - **Output:** `[B, 64]` — temporal summary per entity
  - **Parameters:** ~74K params (~300KB)
  - **Design rationale:**
    - hidden=64 (không phải 128/256) vì chỉ cần bắt short-term changes
    - 1 layer vì latency budget chỉ có 1-2ms
    - Stateful option: giữ hidden state giữa các frame để tiết kiệm recompute
  → Verify: Forward pass <2ms, output shape `[B, 64]`, gradient flows

- [ ] **T2.3**: Design & implement `models/tcn_pathway.py` — Custom Temporal Convolutional Network
  - **Architecture:** 3-block causal dilated CNN
    - Block structure: `DilatedCausalConv1d → BatchNorm → ReLU → Dropout(0.1) → Residual`
    - Dilation pattern: `[1, 2, 4]` → receptive field = 15 timesteps
    - Channels: `128 → 128 → 64 → 64`
    - Kernel size: 3
  - **Input:** `[B, 128, T]` — entity features (channel-first for Conv1d)
  - **Output:** `[B, 64]` — long-term temporal summary (take last timestep)
  - **Parameters:** ~115K params (~460KB)
  - **Design rationale:**
    - Dilated convolutions: bắt được pattern dài (receptive field = 15 steps ≈ 5-7 giây @2-3 Hz tracker) mà KHÔNG tăng params
    - Causal: chỉ nhìn quá khứ, phù hợp real-time
    - 3 block (không phải 5-6): cân bằng giữa receptive field và compute
    - Residual connections: ổn định gradient cho mạng sâu hơn GRU
    - Output cùng dim=64 như GRU để fusion layer thống nhất
  → Verify: Forward pass <3ms, causal property (future input không ảnh hưởng past output), receptive field = 15

- [ ] **T2.4**: Design & implement `models/attention_pathway.py` — Custom Multi-Head Self-Attention
  - **Architecture:** 1-layer Multi-Head Attention
    - Heads: 4, dim_per_head: 32, total dim: 128
    - Pre-LayerNorm + feedforward `[128 → 256 → 128]`
  - **Input:** `[B, N, 128]` — N entities, mỗi entity 128-dim features
  - **Output:** `[B, 128]` — context-aware entity representation (mean-pooled)
  - **Parameters:** ~100K params (~400KB)
  - **Design rationale:**
    - 4 heads: mỗi head học một loại relationship khác nhau (spatial proximity, velocity similarity, appearance, interaction)
    - Mean pool output: vì N (số entities) thay đổi mỗi frame
    - 1 layer: đủ cho entity-level attention, thêm layer sẽ vượt latency budget
  → Verify: Forward pass <8ms, attention weights interpretable (visualize), variable N works

- [ ] **T2.5**: Design & implement `models/gnn_pathway.py` — Custom Graph Attention Network (GAT)
  - **Architecture:** 2-layer GAT
    - Layer 1: `128 → 128`, 4 attention heads, concat → 512 → linear → 128
    - Layer 2: `128 → 256`, 1 attention head
    - Graph construction: edge nếu khoảng cách Euclidean giữa 2 entities < threshold (2.0m)
    - Self-loops included
  - **Input:** `[B, N, 128]` + adjacency matrix `[N, N]` (built from 3D positions)
  - **Output:** `[B, 256]` — graph-level representation (mean pool over nodes)
  - **Parameters:** ~165K params (~660KB)
  - **Design rationale:**
    - GAT (không phải GCN): attention mechanism cho phép mạng tự học edge nào quan trọng
    - 2 layers: đủ để thông tin truyền qua 2-hop neighbors
    - Distance-based graph: phản ánh thực tế — người ở gần nhau mới tương tác
    - Output 256-dim: GNN cần dim lớn hơn vì encode cả structure lẫn features
  → Verify: Forward pass <15ms, graph construction <2ms, handles N=0..20 entities

- [ ] **T2.6**: Design & implement `models/gated_fusion.py` — Gated Fusion Layer
  - **Architecture:** Gated linear combination
    - Gate network: MLP `[pathway_dims_sum → 128 → num_active_pathways]` + Softmax
    - Projection: linear `[pathway_dims_sum → 256]`
    - Active pathways vary: GRU(64) + TCN(64) + Attention(128) + GNN(256) = max 512
  - **Input:** Concatenated outputs từ active pathways (variable-length)
  - **Output:** `[B, 256]` — unified reasoning vector
  - **Parameters:** ~70K params (~280KB)
  - **Design rationale:**
    - Gated (không phải simple average): mạng tự học pathway nào đáng tin hơn cho tình huống hiện tại
    - Softmax gates: weights luôn sum to 1, interpretable
    - Single projection layer: giữ latency thấp
    - Variable input: xử lý được khi Router chỉ bật 1-2 pathways
  → Verify: Weights sum to 1, output dim=256 consistent, gradient flows through gates

### Phần B: System Integration

- [ ] **T2.7**: Implement `src/complexity/estimator.py` — Wrapper cho Complexity Estimator
  - Load trained model từ `models/onnx/estimator.onnx`
  - Extract scene metrics từ perception output → build 36-dim input vector
  → Verify: Correct classification trên synthetic test cases

- [ ] **T2.8**: Implement `src/complexity/soh_monitor.py` — GPU temp, RAM, utilization → soh_budget
  - soh_budget ∈ [0.0, 1.0]: 1.0 = healthy, 0.0 = critical
  - Reads từ `tegrastats` (Jetson) hoặc mock (CI)
  → Verify: soh_budget giảm khi stress-test GPU

- [ ] **T2.9**: Implement `src/router/adaptive_router.py` — Rule-based version
  - Input: complexity_level + soh_budget
  - Output: set of active pathways + latency budget allocation
  - Logic: lookup table theo Routing Strategy ở trên
  - SoH override: nếu soh_budget < 0.5 → force downgrade 1 level
  → Verify: LOW → GRU_ONLY, HIGH → GRU+TCN+ATTN, CRITICAL → ALL, SoH override works

- [ ] **T2.10**: Implement `src/reasoning/` — Pathway inference wrappers
  - Mỗi file load ONNX/TensorRT model tương ứng
  - `gru_pathway.py`, `tcn_pathway.py`, `attention_pathway.py`, `gnn_pathway.py`
  - Lazy loading: chỉ load model khi Router activate pathway đó
  → Verify: Lazy loading works (RAM chỉ tăng khi pathway activated)

- [ ] **T2.11**: Implement `src/reasoning/fusion.py` — Gated Fusion inference wrapper
  → Verify: Output dim=256 consistent bất kể bao nhiêu pathway active

### Phần C: Training Pipeline (Desktop GPU)

- [ ] **T2.12**: Implement `pipelines/train_estimator.py` — Train Complexity Estimator
  - Dataset: synthetic scenes với labeled complexity levels
  - Loss: CrossEntropy
  - Export: PyTorch → ONNX → (later) TensorRT
  → Verify: Accuracy ≥90% trên validation set

- [ ] **T2.13**: Implement `pipelines/train_reasoning.py` — Joint training cho 4 pathways + fusion
  - **Training strategy:** End-to-end, tất cả pathways train cùng lúc
  - Dataset: sequences of entity features (from Phase 1 perception hoặc synthetic)
  - Loss: multi-task loss
    - Activity classification (CE)
    - Direction prediction (CE)
    - Anomaly score (BCE)
    - Auxiliary: reconstruction loss cho representation quality
  - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
  - Export: mỗi pathway → riêng ONNX file
  → Verify: Loss converges, mỗi pathway export thành công

- [ ] **T2.14**: ONNX export script — `scripts/export_brain_onnx.py`
  - Export 6 models: estimator, gru, tcn, attention, gnn, fusion
  - Validate: ONNX checker + shape verification
  → Verify: Tất cả 6 file `.onnx` valid, inference output khớp PyTorch

### Phần D: Testing & Integration

- [ ] **T2.15**: Unit tests cho từng custom network
  - Test forward pass shape
  - Test gradient flow (backward pass không error)
  - Test edge cases: N=0 entities, T=1 timestep, single entity
  → Verify: `pytest tests/unit/test_brain_networks.py` — all pass

- [ ] **T2.16**: Integration test — full brain pipeline
  - Perception output → Estimator → Router → Pathway(s) → Fusion → unified_output
  - Test với 4 complexity levels
  - Measure end-to-end latency per level
  → Verify: LOW <5ms, MED <10ms, HIGH <20ms, CRITICAL <35ms

- [ ] **T2.17**: Update benchmark baselines
  - Add TCN to `latency_baseline.json` và `output_reference/tcn_ref.npy`
  - Update `baseline_meta.json` with new model list
  → Verify: `python scripts/benchmark.py --device ci --compare-baseline` passes

---

## Memory Budget (Jetson 8GB Shared RAM)

| Component | Params | Size (FP16) | Loaded When |
|-----------|--------|-------------|-------------|
| Complexity Estimator | ~3.5K | ~7KB | Always |
| GRU Pathway | ~74K | ~148KB | LOW+ |
| TCN Pathway | ~115K | ~230KB | MED+ |
| Attention Pathway | ~100K | ~200KB | HIGH+ |
| GNN Pathway | ~165K | ~330KB | CRITICAL |
| Gated Fusion | ~70K | ~140KB | Always |
| **Total Brain** | **~527K** | **~1.06MB** | — |

> **Kết luận:** Toàn bộ "bộ não" chỉ chiếm **~1MB GPU RAM (FP16)**. Rất nhỏ so với YOLOv8-s (~14MB). Lazy loading thêm tiết kiệm ~530KB khi ở LOW mode.

---

## File Structure Update

```
src/reasoning/              # Pathway inference wrappers (runtime)
├── gru_pathway.py
├── tcn_pathway.py          # ← NEW
├── attention_pathway.py
├── gnn_pathway.py
└── fusion.py

models/                     # Custom network definitions (PyTorch)
├── complexity_estimator.py
├── gru_pathway.py
├── tcn_pathway.py          # ← NEW
├── attention_pathway.py
├── gnn_pathway.py
├── gated_fusion.py
├── onnx/                   # Exported ONNX files
│   ├── estimator.onnx
│   ├── gru.onnx
│   ├── tcn.onnx            # ← NEW
│   ├── attention.onnx
│   ├── gnn.onnx
│   └── fusion.onnx
└── engines/                # TensorRT engines (built on Jetson, cached)

pipelines/                  # Training scripts (desktop GPU)
├── train_estimator.py
├── train_reasoning.py      # Joint training for all 4 pathways + fusion
├── train_gnn.py            # (optional) standalone GNN training
└── train_attention.py      # (optional) standalone attention training
```

---

## Done When

- [ ] Tất cả 6 custom networks (`nn.Module`) defined, forward pass works
- [ ] Training pipeline runs, loss converges
- [ ] ONNX export thành công cho tất cả 6 models
- [ ] Adaptive routing works: LOW scene <5ms, MED <10ms, HIGH <20ms, CRITICAL <35ms
- [ ] SoH-aware: GPU hot → auto-downgrade pathway
- [ ] Lazy loading: RAM chỉ tăng khi pathway activated
- [ ] Unit + integration tests pass
- [ ] Benchmark baselines updated (thêm TCN)

---

## Specs Updates Required

> Các thay đổi cần đồng bộ sang spec documents:

| Spec File | Change |
|-----------|--------|
| `data-schema.md` §5 | Thêm TCN Pathway I/O contract: `[B, 128, T]` → `[B, 64]`, latency <3ms |
| `benchmarking.md` §2 | Thêm `tcn_pathway` vào CI baseline models |
| `benchmarking.md` §3 | Cập nhật latency measurement modules list |
| Master plan (file structure) | Thêm `tcn_pathway.py` vào `src/reasoning/` và `models/` |
| Master plan (Phase 4) | Update RL action space: `{GRU_ONLY, GRU_TCN, GRU_TCN_ATTN, ALL}` |
