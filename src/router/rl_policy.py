from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.complexity.estimator import ComplexityLevel
from src.runtime.tensorrt_engine import TensorRTEngineRunner
from src.utils.math import clip01, softmax


class RoutingAction(IntEnum):
    FAST = 0
    BALANCED = 1
    ACCURATE = 2
    EMERGENCY = 3


@dataclass(frozen=True, slots=True)
class RLRouterState:
    crowd_density: float
    motion_entropy: float
    anomaly_probability: float
    soh_budget: float
    scene_embedding: np.ndarray
    prev_action: RoutingAction = RoutingAction.FAST
    time_since_critical: float = 1.0
    current_fps: float = 1.0

    INPUT_DIM = 39
    SCENE_EMBEDDING_DIM = 32

    def to_vector(self) -> np.ndarray:
        scene_embedding = np.asarray(self.scene_embedding, dtype=np.float32)
        if scene_embedding.shape != (self.SCENE_EMBEDDING_DIM,):
            raise ValueError("scene_embedding must have shape [32]")
        return np.concatenate(
            [
                np.array(
                    [
                        clip01(self.crowd_density),
                        clip01(self.motion_entropy),
                        clip01(self.anomaly_probability),
                        clip01(self.soh_budget),
                    ],
                    dtype=np.float32,
                ),
                scene_embedding,
                np.array(
                    [
                        float(self.prev_action) / len(RoutingAction),
                        clip01(self.time_since_critical),
                        clip01(self.current_fps),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).reshape(1, self.INPUT_DIM)

    @classmethod
    def from_complexity(
        cls,
        complexity_level: ComplexityLevel | int,
        *,
        soh_budget: float,
        anomaly_probability: float = 0.0,
        scene_embedding: np.ndarray | None = None,
        prev_action: RoutingAction = RoutingAction.FAST,
        time_since_critical: float = 1.0,
        current_fps: float = 1.0,
    ) -> "RLRouterState":
        level = ComplexityLevel(complexity_level)
        density_map = {
            ComplexityLevel.LOW: 0.15,
            ComplexityLevel.MED: 0.40,
            ComplexityLevel.HIGH: 0.70,
            ComplexityLevel.CRITICAL: 0.95,
        }
        entropy_map = {
            ComplexityLevel.LOW: 0.10,
            ComplexityLevel.MED: 0.30,
            ComplexityLevel.HIGH: 0.65,
            ComplexityLevel.CRITICAL: 0.95,
        }
        embedding = (
            np.asarray(scene_embedding, dtype=np.float32)
            if scene_embedding is not None
            else np.full((cls.SCENE_EMBEDDING_DIM,), float(level) / 3.0, dtype=np.float32)
        )
        return cls(
            crowd_density=density_map[level],
            motion_entropy=entropy_map[level],
            anomaly_probability=anomaly_probability,
            soh_budget=soh_budget,
            scene_embedding=embedding,
            prev_action=prev_action,
            time_since_critical=time_since_critical,
            current_fps=current_fps,
        )


@dataclass(frozen=True, slots=True)
class RLPolicyDecision:
    action: RoutingAction
    logits: np.ndarray
    probabilities: np.ndarray
    state_vector: np.ndarray


class RLPolicyRuntime(Protocol):
    def run(self, input_batch: np.ndarray) -> np.ndarray:
        """Return logits for a `[B, 39]` float32 input batch."""


class RLPolicy:
    DEFAULT_MODEL_PATH = Path("models/engines/rl_policy.engine")

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        runtime: RLPolicyRuntime | None = None,
    ) -> None:
        self.runtime = runtime or _TensorRTRouterPolicyRuntime(Path(model_path))

    def decide(self, state: RLRouterState) -> RLPolicyDecision:
        state_vector = state.to_vector().astype(np.float32, copy=False)
        logits = np.asarray(self.runtime.run(state_vector), dtype=np.float32)
        if logits.shape != (state_vector.shape[0], len(RoutingAction)):
            raise ValueError("rl policy runtime must return logits with shape [B, 4]")
        probabilities = softmax(logits)
        action = RoutingAction(int(np.argmax(probabilities[0])))
        return RLPolicyDecision(
            action=action,
            logits=logits[0].copy(),
            probabilities=probabilities[0].copy(),
            state_vector=state_vector[0].copy(),
        )


class _TensorRTRouterPolicyRuntime:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"rl policy TensorRT engine not found: {model_path}")
        self.runner = TensorRTEngineRunner(model_path, ("router_state",))

    def run(self, input_batch: np.ndarray) -> np.ndarray:
        return self.runner.run(input_batch)


@dataclass(frozen=True, slots=True)
class RLScenario:
    scenario_id: int
    crowd_density: float
    motion_entropy: float
    expected_action: RoutingAction
    anomaly_probability: float = 0.0
    soh_budget: float = 1.0
    time_since_critical: float = 1.0
    current_fps: float = 1.0
    scene_embedding: np.ndarray | None = None

    def to_state(self, prev_action: RoutingAction = RoutingAction.FAST) -> RLRouterState:
        embedding = (
            np.asarray(self.scene_embedding, dtype=np.float32)
            if self.scene_embedding is not None
            else _scenario_embedding(
                self.crowd_density,
                self.motion_entropy,
                self.anomaly_probability,
                self.soh_budget,
            )
        )
        return RLRouterState(
            crowd_density=self.crowd_density,
            motion_entropy=self.motion_entropy,
            anomaly_probability=self.anomaly_probability,
            soh_budget=self.soh_budget,
            scene_embedding=embedding,
            prev_action=prev_action,
            time_since_critical=self.time_since_critical,
            current_fps=self.current_fps,
        )


class AdaptiveRoutingEnv:
    def __init__(self, scenarios: list[RLScenario]) -> None:
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        self.scenarios = scenarios
        self.index = 0
        self.prev_action = RoutingAction.FAST
        self.steps = 0
        try:
            from gymnasium import spaces
        except ImportError:
            self.observation_space = None
            self.action_space = None
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(39,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(RoutingAction))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self.index = 0
        self.steps = 0
        self.prev_action = RoutingAction.FAST
        state = self.scenarios[self.index].to_state(self.prev_action).to_vector()[0]
        return state.astype(np.float32, copy=False), {"scenario_id": self.scenarios[self.index].scenario_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        scenario = self.scenarios[self.index]
        selected_action = RoutingAction(action)
        reward, info = evaluate_action(selected_action, scenario)
        self.prev_action = selected_action
        self.steps += 1
        terminated = self.index >= len(self.scenarios) - 1
        if not terminated:
            self.index += 1
        next_state = self.scenarios[self.index].to_state(self.prev_action).to_vector()[0]
        info = {
            **info,
            "scenario_id": scenario.scenario_id,
            "expected_action": scenario.expected_action.name,
            "selected_action": selected_action.name,
        }
        return next_state.astype(np.float32, copy=False), reward, terminated, False, info

    @classmethod
    def from_fixture_dir(cls, fixture_dir: str | Path) -> "AdaptiveRoutingEnv":
        fixture_path = Path(fixture_dir)
        scenarios = [load_rl_scenario(path) for path in sorted(fixture_path.glob("scenario_*.json"))]
        return cls(scenarios)


def load_rl_scenario(path: str | Path) -> RLScenario:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RLScenario(
        scenario_id=int(payload["scenario_id"]),
        crowd_density=float(payload["crowd_density"]),
        motion_entropy=float(payload["motion_entropy"]),
        anomaly_probability=float(payload.get("anomaly_probability", 0.0)),
        soh_budget=float(payload.get("soh_budget", 1.0)),
        expected_action=RoutingAction[str(payload["expected_action"])],
        time_since_critical=float(payload.get("time_since_critical", 1.0)),
        current_fps=float(payload.get("current_fps", 1.0)),
        scene_embedding=np.asarray(
            payload.get(
                "scene_embedding",
                _scenario_embedding(
                    float(payload["crowd_density"]),
                    float(payload["motion_entropy"]),
                    float(payload.get("anomaly_probability", 0.0)),
                    float(payload.get("soh_budget", 1.0)),
                ),
            ),
            dtype=np.float32,
        ),
    )


def evaluate_action(action: RoutingAction, scenario: RLScenario) -> tuple[float, dict[str, float | bool]]:
    action_complexity = (int(action) + 1) / len(RoutingAction)
    target_complexity = (int(scenario.expected_action) + 1) / len(RoutingAction)
    accuracy = 1.0 - abs(action_complexity - target_complexity)
    latency_penalty = 0.08 * int(action)
    energy_penalty = 0.06 * int(action) * max(0.2, 1.0 - scenario.soh_budget)
    anomaly_bonus = (
        0.2
        if scenario.expected_action == RoutingAction.EMERGENCY and action == scenario.expected_action
        else 0.0
    )
    reward = accuracy - latency_penalty - energy_penalty + anomaly_bonus
    return reward, {
        "accuracy": accuracy,
        "latency_penalty": latency_penalty,
        "energy_penalty": energy_penalty,
        "anomaly_bonus": anomaly_bonus,
        "match": action == scenario.expected_action,
    }


def composite_score(actions: list[RoutingAction], scenarios: list[RLScenario]) -> float:
    if len(actions) != len(scenarios):
        raise ValueError("actions and scenarios must have the same length")
    rewards = [evaluate_action(action, scenario)[0] for action, scenario in zip(actions, scenarios, strict=True)]
    return float(np.mean(rewards))


def baseline_rule_action(scenario: RLScenario) -> RoutingAction:
    if scenario.soh_budget < 0.35:
        return RoutingAction.FAST
    if scenario.crowd_density >= 0.85 or scenario.motion_entropy >= 0.85:
        return RoutingAction.ACCURATE
    if scenario.crowd_density >= 0.45 or scenario.motion_entropy >= 0.45:
        return RoutingAction.BALANCED
    return RoutingAction.FAST


def _scenario_embedding(
    crowd_density: float,
    motion_entropy: float,
    anomaly_probability: float,
    soh_budget: float,
) -> np.ndarray:
    base = np.array(
        [
            clip01(crowd_density),
            clip01(motion_entropy),
            clip01(anomaly_probability),
            clip01(soh_budget),
            clip01(0.7 * crowd_density + 0.3 * motion_entropy),
            clip01(max(crowd_density, motion_entropy)),
            clip01(abs(crowd_density - motion_entropy)),
            clip01((crowd_density + motion_entropy + anomaly_probability) / 3.0),
        ],
        dtype=np.float32,
    )
    return np.tile(base, 4)[: RLRouterState.SCENE_EMBEDDING_DIM].astype(np.float32, copy=False)
