# Phase 4: RL Policy for Adaptive Router

> Train PPO policy thay thế rule-based router

## Goal

Thay rule-based router (Phase 2) bằng learned RL policy, tối ưu accuracy-latency-energy tradeoff.

## Tasks

- [ ] **T4.1**: Implement simulation environment cho RL training
  - Gym-compatible env, simulates scenes với varying complexity
  - State: complexity metrics + soh_budget + scene_embedding
  - Action: pathway selection {GRU_ONLY, GRU_ATTN, ATTN_GNN, ALL}
  - Reward: accuracy - latency_penalty - energy_cost + anomaly_bonus
  → Verify: `env.step()` returns valid (state, reward, done, info)

- [ ] **T4.2**: Implement `rl_policy.py` — PPO training script
  - Use Stable-Baselines3
  - Train trên desktop GPU
  → Verify: Reward curve increases over training

- [ ] **T4.3**: Export trained policy → ONNX → TensorRT
  → Verify: Policy inference <1ms trên Jetson

- [ ] **T4.4**: Integrate RL policy vào `adaptive_router.py`
  - Fallback: nếu RL fails → rule-based
  → Verify: RL router outperforms rule-based trên test scenarios

- [ ] **T4.5**: Online adaptation (optional) — lightweight policy fine-tune trên Jetson
  → Verify: Policy adapts to new environment trong <1000 episodes

## Done When

- [ ] RL policy chọn pathway tốt hơn rule-based (measured by composite score)
- [ ] Policy inference <1ms
- [ ] Fallback mechanism works khi RL fails
