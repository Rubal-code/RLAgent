
import argparse
import random
from pathlib import Path

from rl_agent_lab.agents.dqn_model_io import save_model
from rl_agent_lab.agents.model_io import save_q_table
from rl_agent_lab.training.experiment_tracker import log_experiment
from rl_agent_lab.training.train_dqn import train as train_dqn
from rl_agent_lab.training.train_q_learning import train as train_q_learning


def run_q_learning(episodes: int, seed: int) -> None:
    random.seed(seed)

    agent, rewards, successes = train_q_learning(episodes=episodes)

    if not rewards or not successes:
        raise RuntimeError("Q-Learning training returned empty results.")

    model_path = Path("models") / "q_table.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_q_table(agent, str(model_path))

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": sum(rewards) / len(rewards),
        "success_rate": sum(successes) / len(successes),
        "final_100_average_reward": sum(recent_rewards) / len(recent_rewards),
    }

    log_experiment(
        algorithm="Q-Learning",
        environment="GridWorld",
        seed=seed,
        episodes=episodes,
        metrics=metrics,
        model_path=str(model_path),
    )

    print("\n--- Experiment Summary ---")
    print("Algorithm: Q-Learning")
    print("Environment: GridWorld")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {metrics['average_reward']:.2f}")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Model saved to: {model_path}")
    print("Results logged to: results/experiments.jsonl")


def run_dqn(episodes: int, seed: int) -> None:
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, rewards = train_dqn(episodes=episodes)

    if not rewards:
        raise RuntimeError("DQN training returned empty results.")

    model_path = Path("models") / "cartpole_dqn.pth"
    save_model(agent, str(model_path))

    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": sum(rewards) / len(rewards),
        "final_100_average_reward": sum(recent_rewards) / len(recent_rewards),
        "best_reward": max(rewards),
    }

    log_experiment(
        algorithm="DQN",
        environment="CartPole-v1",
        seed=seed,
        episodes=episodes,
        metrics=metrics,
        model_path=str(model_path),
    )

    print("\n--- Experiment Summary ---")
    print("Algorithm: DQN")
    print("Environment: CartPole-v1")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {metrics['average_reward']:.2f}")
    print(f"Final 100 average reward: {metrics['final_100_average_reward']:.2f}")
    print(f"Best reward: {metrics['best_reward']:.0f}")
    print(f"Model saved to: {model_path}")
    print("Results logged to: results/experiments.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RL experiment runner")
    parser.add_argument(
        "--algorithm",
        choices=["q-learning", "dqn"],
        default="q-learning",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    if args.algorithm == "q-learning":
        run_q_learning(args.episodes, args.seed)
    else:
        run_dqn(args.episodes, args.seed)


if __name__ == "__main__":
    main()
