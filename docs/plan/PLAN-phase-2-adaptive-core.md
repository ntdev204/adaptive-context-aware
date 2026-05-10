# Phase 2: Adaptive Core

> Complexity Estimator + Adaptive Router + GRU/Attention/GNN Pathways + Fusion

## Goal

Implement hệ thống AI "nghĩ tỷ lệ thuận với độ phức tạp" — lightweight khi đơn giản, deep khi phức tạp.

## Tasks

- [ ] **T2.1**: Implement `estimator.py` — Complexity Estimator (tiny MLP)
  - Input: crowd_density, motion_entropy, anomaly_score_prev, scene_embedding
  - Output: complexity_level ∈ {LOW=0, MED=1, HIGH=2, CRITICAL=3}
  → Verify: Correct classification trên synthetic test cases

- [ ] **T2.2**: Implement `soh_monitor.py` — GPU temp, RAM, utilization → soh_budget
  → Verify: soh_budget giảm khi stress-test GPU

- [ ] **T2.3**: Implement `gru_pathway.py` — lightweight temporal reasoning (~1ms)
  - 1-layer GRU, hidden=64, input=entity_features
  → Verify: Forward pass <2ms, output shape correct

- [ ] **T2.4**: Implement `attention_pathway.py` — self-attention across entities (~5ms)
  - Multi-head attention (4 heads, dim=128)
  → Verify: Forward pass <8ms, attention weights interpretable

- [ ] **T2.5**: Implement `gnn_pathway.py` — Graph Attention Network (~10ms)
  - 2-layer GAT, build graph from entity spatial proximity
  → Verify: Forward pass <15ms, graph construction correct

- [ ] **T2.6**: Implement `fusion.py` — Gated fusion layer
  - Weighted combination dựa trên router output
  → Verify: Weights sum to 1, output dim correct

- [ ] **T2.7**: Implement `adaptive_router.py` — rule-based version first
  - Thresholds dựa trên complexity_level + soh_budget
  → Verify: LOW → GRU_ONLY, HIGH → ATTENTION_GNN, etc.

- [ ] **T2.8**: Integration test — full pipeline perception → estimator → router → pathway → fusion
  → Verify: End-to-end pass, latency within budget per complexity level

## Done When

- [ ] Adaptive routing works: LOW scene <15ms, HIGH scene <40ms
- [ ] SoH-aware: GPU hot → auto-downgrade pathway
- [ ] Unit + integration tests pass
