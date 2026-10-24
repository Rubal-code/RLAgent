
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from rl_agent_lab.training.experiment_tracker import log_experiment
from rl_agent_lab.training.train_dqn import train as train_dqn
from rl_agent_lab.training.train_reinforce import (
    train as train_reinforce,
)


def calculate_metrics(rewards: list[float]) -> dict[str, float]:
    """Calculate metrics for one training run."""
    if not rewards:
        raise ValueError("Rewards cannot be empty.")

    recent_rewards = rewards[-min(100, len(rewards)):]

    return {
        "average_reward": float(mean(rewards)),
        "final_100_average_reward": float(mean(recent_rewards)),
        "best_reward": float(max(rewards)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Calculate mean and population standard deviation by algorithm."""
    summary = {}

    for algorithm in ("DQN", "REINFORCE"):
        algorithm_runs = [
            run for run in runs if run["algorithm"] == algorithm
        ]

        if not algorithm_runs:
            continue

        summary[algorithm] = {}

        for metric in (
            "average_reward",
            "final_100_average_reward",
            "best_reward",
        ):
            values = [
                run["metrics"][metric] for run in algorithm_runs
            ]

            summary[algorithm][metric] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
            }

    return summary


def run_evaluation(
    episodes: int,
    seeds: list[int],
    output_path: Path,
) -> dict:
    """Evaluate DQN and REINFORCE over multiple seeds and log each run."""
    if episodes <= 0:
        raise ValueError("Episodes must be greater than zero.")

    if not seeds:
        raise ValueError("Provide at least one seed.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")

    trainers = {
        "DQN": train_dqn,
        "REINFORCE": train_reinforce,
    }

    runs = []

    for algorithm, trainer in trainers.items():
        for seed in seeds:
            print(
                f"\nTraining {algorithm}: "
                f"episodes={episodes}, seed={seed}"
            )

            _, rewards = trainer(episodes=episodes, seed=seed)
            metrics = calculate_metrics(rewards)

            # Record this run in the append-only experiment history.
            log_experiment(
                algorithm=algorithm,
                environment="CartPole-v1",
                seed=seed,
                episodes=episodes,
                metrics=metrics,
                results_dir=output_path.parent,
            )

            run = {
                "algorithm": algorithm,
                "environment": "CartPole-v1",
                "episodes": episodes,
                "seed": seed,
                "metrics": metrics,
            }
            runs.append(run)

            print(
                f"Average reward: {metrics['average_reward']:.2f} | "
                f"Final-100 average: "
                f"{metrics['final_100_average_reward']:.2f}"
            )

    result = {
        "episodes_per_run": episodes,
        "seeds": seeds,
        "runs": runs,
        "summary": summarize_runs(runs),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n--- Multi-Seed Evaluation Summary ---")

    for algorithm, metrics in result["summary"].items():
        print(f"\n{algorithm}")

        for metric, values in metrics.items():
            print(
                f"  {metric}: "
                f"{values['mean']:.2f} "
                f"+/- {values['std']:.2f}"
            )

    print(f"\nFull results saved to: {output_path}")
    print(
        "Individual runs logged to: "
        f"{output_path.parent / 'experiments.jsonl'}"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DQN and REINFORCE across multiple seeds."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multi_seed_evaluation.json"),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")

    run_evaluation(
        episodes=args.episodes,
        seeds=args.seeds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
