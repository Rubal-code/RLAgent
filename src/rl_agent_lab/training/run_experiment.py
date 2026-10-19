import argparse
import random
from pathlib import Path

from rl_agent_lab.agents.model_io import save_q_table
from rl_agent_lab.training.experiment_tracker import log_experiment
from rl_agent_lab.training.train_q_learning import train


def run_q_learning(episodes: int, seed: int) -> None:
    """Train Q-Learning, save the model, and record experiment metrics."""
    random.seed(seed)

    agent, rewards, successes = train(episodes=episodes)

    if not rewards or not successes:
        raise RuntimeError("Training returned no results.")

    # Save the newly trained Q-table.
    model_path = Path("models") / "q_table.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    save_q_table(agent, str(model_path))

    average_reward = sum(rewards) / len(rewards)
    success_rate = sum(successes) / len(successes)
    recent_rewards = rewards[-min(100, len(rewards)):]

    metrics = {
        "average_reward": average_reward,
        "success_rate": success_rate,
        "final_100_average_reward": (
            sum(recent_rewards) / len(recent_rewards)
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
    print("Algorithm: Q-Learning")
    print("Environment: GridWorld")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Model saved to: {model_path}")
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