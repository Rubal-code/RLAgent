
import argparse
from pathlib import Path

from rl_agent_lab.training.experiment_tracker import log_experiment
from rl_agent_lab.training.train_q_learning import train


def run_q_learning(episodes: int, seed: int) -> None:
    """Train Q-Learning on GridWorld and record its metrics."""
    import random

    random.seed(seed)

    agent, rewards, successes = train(episodes=episodes)

    average_reward = sum(rewards) / len(rewards)
    success_rate = sum(successes) / len(successes)

    model_path = Path("models") / "q_table.json"

    metrics = {
        "average_reward": average_reward,
        "success_rate": success_rate,
        "final_100_average_reward": (
            sum(rewards[-100:]) / len(rewards[-100:])
        ),
    }

    log_path = log_experiment(
        algorithm="Q-Learning",
        environment="GridWorld",
        seed=seed,
        episodes=episodes,
        metrics=metrics,
        model_path=str(model_path),
    )

    print("\n--- Experiment Summary ---")
    print(f"Algorithm: Q-Learning")
    print(f"Environment: GridWorld")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Results logged to: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and record a reinforcement learning experiment."
    )
    parser.add_argument(
        "--algorithm",
        choices=["q-learning"],
        default="q-learning",
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero.")

    run_q_learning(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
