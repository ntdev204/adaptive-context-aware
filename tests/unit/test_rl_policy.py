from __future__ import annotations

import numpy as np

from src.router.rl_policy import AdaptiveRoutingEnv, RLPolicy, RLRouterState, RoutingAction, composite_score


def test_rl_router_state_matches_contract_shape() -> None:
    state = RLRouterState(
        crowd_density=0.4,
        motion_entropy=0.6,
        anomaly_probability=0.2,
        soh_budget=0.8,
        scene_embedding=np.zeros(32, dtype=np.float32),
        prev_action=RoutingAction.BALANCED,
        time_since_critical=0.5,
        current_fps=0.75,
    )

    vector = state.to_vector()

    assert vector.shape == (1, 39)
    assert vector.dtype == np.float32


def test_rl_env_step_returns_gym_style_tuple() -> None:
    env = AdaptiveRoutingEnv.from_fixture_dir("tests/fixtures/rl_scenarios")

    state, info = env.reset()
    next_state, reward, terminated, truncated, step_info = env.step(RoutingAction.FAST)

    assert state.shape == (39,)
    assert "scenario_id" in info
    assert next_state.shape == (39,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "expected_action" in step_info


def test_rl_policy_decodes_action_from_logits() -> None:
    class FakeRuntime:
        def run(self, input_batch):
            return np.array([[0.0, 0.1, 0.2, 1.4]], dtype=np.float32)

    policy = RLPolicy(runtime=FakeRuntime())
    decision = policy.decide(RLRouterState.from_complexity(1, soh_budget=1.0))

    assert decision.action == RoutingAction.EMERGENCY
    assert decision.probabilities.shape == (4,)


def test_rule_based_reference_score_matches_fixture_count() -> None:
    env = AdaptiveRoutingEnv.from_fixture_dir("tests/fixtures/rl_scenarios")
    actions = [
        RoutingAction.FAST,
        RoutingAction.ACCURATE,
        RoutingAction.BALANCED,
        RoutingAction.ACCURATE,
        RoutingAction.EMERGENCY,
    ]

    score = composite_score(actions, env.scenarios)

    assert score > 0.7
