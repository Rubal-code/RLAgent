
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from rl_agent_lab.agents.dqn_model_io import save_model
from rl_agent_lab.agents.model_io import save_q_table
from rl_agent_lab.agents.reinforce_model_io import (
    save_model as save_reinforce_model,
)
from rl_agent_lab.training.experiment_tracker import log_experiment
from rl_agent_lab.training.train_dqn import train as train_dqn
from rl_agent_lab.training.train_ppo import train as train_ppo
from rl_agent_lab.training.train_q_learning import train as train_q_learning
from rl_agent_lab.training.train_reinforce import train as train_reinforce


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_summary(
    algorithm: str,
    environment: str,
    episodes: int,
    seed: int,
    metrics: dict[str, float],
    model_path: Path,
) -> None:
    """Log experiment metadata and print a concise summary."""
    log_experiment(
        algorithm=algorithm,
        environment=environment,
        seed=seed,
        episodes=episodes,
        metrics=metrics,
        model_path=str(model_path),
    )

    print("\n--- Experiment Summary ---")
    print(f"Algorithm: {algorithm}")
    print(f"Environment: {environment}")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed}")

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.3f}")

    print(f"Model saved to: {model_path}")
    print("Results logged to: results/experiments.jsonl")


def run_q_learning(episodes: int, seed: int) -> None:
    set_seed(seed)
    agent, rewards, successes = train_q_learning(episodes=episodes)

    if not rewards or not successes:
        raise RuntimeError("Q-Learning training returned empty results.")

    model_path = Path("models") / "q_table.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_q_table(agent, str(model_path))

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "final_100_average_reward": float(np.mean(recent_rewards)),
    }

    log_summary(
        "Q-Learning", "GridWorld", episodes, seed, metrics, model_path
    )


def run_dqn(episodes: int, seed: int) -> None:
    set_seed(seed)
    agent, rewards = train_dqn(episodes=episodes)

    if not rewards:
        raise RuntimeError("DQN training returned empty results.")

    model_path = Path("models") / "cartpole_dqn.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(agent, str(model_path))

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": float(np.mean(rewards)),
        "final_100_average_reward": float(np.mean(recent_rewards)),
        "best_reward": float(max(rewards)),
    }

    log_summary(
        "DQN", "CartPole-v1", episodes, seed, metrics, model_path
    )


def run_reinforce(episodes: int, seed: int) -> None:
    set_seed(seed)
    agent, rewards = train_reinforce(episodes=episodes)

    if not rewards:
        raise RuntimeError("REINFORCE training returned empty results.")

    model_path = Path("models") / "cartpole_reinforce.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_reinforce_model(agent, str(model_path))

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": float(np.mean(rewards)),
        "final_100_average_reward": float(np.mean(recent_rewards)),
        "best_reward": float(max(rewards)),
    }

    log_summary(
        "REINFORCE", "CartPole-v1", episodes, seed, metrics, model_path
    )


def run_ppo(episodes: int, seed: int) -> None:
    set_seed(seed)
    agent, rewards = train_ppo(
    episodes=episodes,
    seed=seed,
    resume=False,
)

    if not rewards:
        raise RuntimeError("PPO training returned empty results.")

    model_path = Path("models") / "lunarlander_ppo.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.policy.state_dict(), model_path)

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": float(np.mean(rewards)),
        "final_100_average_reward": float(np.mean(recent_rewards)),
        "best_reward": float(max(rewards)),
    }

    log_summary(
        "PPO", "LunarLander-v3", episodes, seed, metrics, model_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified RL experiment runner"
    )
    parser.add_argument(
        "--algorithm",
        choices=["q-learning", "dqn", "reinforce", "ppo"],
        default="q-learning",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    runners = {
        "q-learning": run_q_learning,
        "dqn": run_dqn,
        "reinforce": run_reinforce,
        "ppo": run_ppo,
    }

    runners[args.algorithm](args.episodes, args.seed)


if __name__ == "__main__":
    main()
