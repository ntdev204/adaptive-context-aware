from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.rl_policy import RLPolicyNet
from src.router.rl_policy import AdaptiveRoutingEnv, RoutingAction, baseline_rule_action, composite_score


@dataclass(frozen=True, slots=True)
class RLTrainingConfig:
    fixture_dir: Path = Path("tests/fixtures/rl_scenarios")
    epochs: int = 160
    batch_size: int = 16
    learning_rate: float = 0.02
    validation_fraction: float = 0.2
    seed: int = 17
    checkpoint_path: Path = Path("models/checkpoints/rl_policy.pt")
    backend: str = "auto"


@dataclass(frozen=True, slots=True)
class RLTrainingResult:
    backend: str
    validation_accuracy: float
    reward_before: float
    reward_after: float
    checkpoint_path: Path | None


def generate_router_training_dataset(
    fixture_dir: Path = Path("tests/fixtures/rl_scenarios"),
) -> tuple[torch.Tensor, torch.Tensor]:
    env = AdaptiveRoutingEnv.from_fixture_dir(fixture_dir)
    states: list[np.ndarray] = []
    labels: list[int] = []
    for scenario in env.scenarios:
        for prev_action in RoutingAction:
            states.append(scenario.to_state(prev_action).to_vector()[0])
            labels.append(int(scenario.expected_action))
    return torch.tensor(np.stack(states), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def train_router_policy(config: RLTrainingConfig = RLTrainingConfig()) -> RLTrainingResult:
    features, labels = generate_router_training_dataset(config.fixture_dir)
    train_dataset, validation_dataset = _split_dataset(features, labels, config.validation_fraction, config.seed)
    model = RLPolicyNet()

    backend = "ppo" if config.backend == "ppo" or (config.backend == "auto" and _has_stable_baselines()) else "supervised"
    if backend == "ppo":
        _warm_start_with_ppo(model, config.fixture_dir, seed=config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    torch.manual_seed(config.seed)
    for _ in range(config.epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    validation_accuracy = _evaluate_accuracy(model, validation_dataset)
    checkpoint_path = _save_checkpoint(model, config.checkpoint_path, backend)
    reward_before = _rule_based_reward(config.fixture_dir)
    reward_after = _trained_policy_reward(model, config.fixture_dir)
    return RLTrainingResult(
        backend=backend,
        validation_accuracy=validation_accuracy,
        reward_before=reward_before,
        reward_after=reward_after,
        checkpoint_path=checkpoint_path,
    )


def _has_stable_baselines() -> bool:
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True


def _warm_start_with_ppo(model: RLPolicyNet, fixture_dir: Path, *, seed: int) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    scenarios_env = AdaptiveRoutingEnv.from_fixture_dir(fixture_dir)

    def make_env() -> AdaptiveRoutingEnv:
        return AdaptiveRoutingEnv(list(scenarios_env.scenarios))

    vec_env = make_vec_env(make_env, n_envs=1, seed=seed)
    policy = PPO("MlpPolicy", vec_env, seed=seed, verbose=0, n_steps=16, batch_size=16, learning_rate=3e-4)
    policy.learn(total_timesteps=256)

    policy_state = policy.policy.state_dict()
    model_state = model.state_dict()
    compatible_keys = {
        key: value
        for key, value in policy_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model_state.update(compatible_keys)
    model.load_state_dict(model_state)


def _split_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    validation_size = max(1, int(features.shape[0] * validation_fraction))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    return (
        TensorDataset(features[train_indices], labels[train_indices]),
        TensorDataset(features[validation_indices], labels[validation_indices]),
    )


def _evaluate_accuracy(model: RLPolicyNet, dataset: TensorDataset) -> float:
    loader = DataLoader(dataset, batch_size=128)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            predictions = torch.argmax(model(features), dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise ValueError("validation dataset must not be empty")
    return correct / total


def _rule_based_reward(fixture_dir: Path) -> float:
    env = AdaptiveRoutingEnv.from_fixture_dir(fixture_dir)
    actions = [baseline_rule_action(scenario) for scenario in env.scenarios]
    return composite_score(actions, env.scenarios)


def _trained_policy_reward(model: RLPolicyNet, fixture_dir: Path) -> float:
    env = AdaptiveRoutingEnv.from_fixture_dir(fixture_dir)
    actions: list[RoutingAction] = []
    model.eval()
    with torch.no_grad():
        for scenario in env.scenarios:
            state = torch.tensor(scenario.to_state().to_vector(), dtype=torch.float32)
            action = RoutingAction(int(torch.argmax(model(state), dim=1).item()))
            actions.append(action)
    return composite_score(actions, env.scenarios)


def _save_checkpoint(model: RLPolicyNet, checkpoint_path: Path, backend: str) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 39,
            "output_dim": 4,
            "backend": backend,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    default_config = RLTrainingConfig()
    parser = argparse.ArgumentParser(description="Train the Phase 4 RL router policy checkpoint.")
    parser.add_argument("--fixture-dir", type=Path, default=default_config.fixture_dir)
    parser.add_argument("--epochs", type=int, default=default_config.epochs)
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=default_config.learning_rate)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--backend", choices=("auto", "ppo", "supervised"), default=default_config.backend)
    parser.add_argument("--checkpoint-path", type=Path, default=default_config.checkpoint_path)
    args = parser.parse_args()

    result = train_router_policy(
        RLTrainingConfig(
            fixture_dir=args.fixture_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            checkpoint_path=args.checkpoint_path,
            backend=args.backend,
        )
    )
    print(f"backend={result.backend}")
    print(f"validation_accuracy={result.validation_accuracy:.4f}")
    print(f"reward_before={result.reward_before:.4f}")
    print(f"reward_after={result.reward_after:.4f}")
    print(f"checkpoint_path={result.checkpoint_path}")


if __name__ == "__main__":
    main()
