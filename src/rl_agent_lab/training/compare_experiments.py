
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_experiments(path: str | Path) -> list[dict[str, Any]]:
    """Load historical experiment records from a JSONL file."""
    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Experiment history not found: {input_path}"
        )

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"in {input_path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}"
                )

            records.append(record)

    return records


def compare_experiments(
    records: list[dict[str, Any]],
    *,
    metric: str = "average_reward",
    environment: str | None = None,
    algorithms: list[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Compare algorithms using metrics from historical runs."""
    if not metric.strip():
        raise ValueError("Metric must not be empty.")

    selected_algorithms = (
        set(algorithms) if algorithms is not None else None
    )
    grouped_values: dict[str, list[float]] = {}

    for record in records:
        if environment is not None:
            if record.get("environment") != environment:
                continue

        algorithm = record.get("algorithm")

        if not isinstance(algorithm, str) or not algorithm.strip():
            continue

        if (
            selected_algorithms is not None
            and algorithm not in selected_algorithms
        ):
            continue

        metrics = record.get("metrics")

        if not isinstance(metrics, dict) or metric not in metrics:
            continue

        value = metrics[metric]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue

        grouped_values.setdefault(algorithm, []).append(float(value))

    summary = {}

    for algorithm, values in sorted(grouped_values.items()):
        summary[algorithm] = {
            "count": len(values),
            "mean": float(mean(values)),
            "std": float(pstdev(values)),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare historical RL experiments."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("results/experiments.jsonl"),
        help="Path to the experiment JSONL file.",
    )
    parser.add_argument(
        "--environment",
        help="Only compare experiments from this environment.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        help="Algorithms to compare, e.g. DQN REINFORCE.",
    )
    parser.add_argument(
        "--metric",
        default="average_reward",
        help="Metric to compare (default: average_reward).",
    )

    args = parser.parse_args()

    try:
        records = load_experiments(args.file)
        summary = compare_experiments(
            records,
            metric=args.metric,
            environment=args.environment,
            algorithms=args.algorithms,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not summary:
        print("No matching experiment records found.")
        return

    print(f"\nExperiment comparison: {args.metric}")
    print("-" * 68)
    print(f"{'Algorithm':<18}{'Runs':>8}{'Mean':>18}{'Std. dev.':>18}")
    print("-" * 68)

    for algorithm, values in summary.items():
        print(
            f"{algorithm:<18}"
            f"{values['count']:>8}"
            f"{values['mean']:>18.2f}"
            f"{values['std']:>18.2f}"
        )


if __name__ == "__main__":
    main()
