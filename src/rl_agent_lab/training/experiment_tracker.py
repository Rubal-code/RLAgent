
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Increment this whenever the record schema has a breaking change.
SCHEMA_VERSION = 1


def _validate_metrics(metrics: dict[str, Any]) -> None:
    """Raise if any metric value is non-finite or non-serialisable."""
    import math

    for key, value in metrics.items():
        if isinstance(value, bool):
            raise TypeError(
                f"Metric '{key}' is a bool; use a numeric value instead."
            )
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(
                    f"Metric '{key}' is non-finite ({value!r}); "
                    "NaN and infinity are not allowed."
                )


def log_experiment(
    *,
    algorithm: str,
    environment: str,
    seed: int,
    episodes: int,
    metrics: dict[str, Any],
    model_path: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    status: str = "completed",
    experiment_id: str | None = None,
    results_dir: str | Path = "results",
) -> Path:
    """Append a single experiment record to the JSONL history file.

    Args:
        algorithm: Name of the RL algorithm (e.g. 'DQN').
        environment: Gymnasium / custom environment identifier.
        seed: Random seed used for this run.
        episodes: Number of training episodes completed.
        metrics: Dictionary of numeric outcome metrics. NaN and infinity
            are rejected. Boolean values are rejected.
        model_path: Optional path where the trained model was saved.
        hyperparameters: Optional dictionary of training hyperparameters.
        status: Execution status. Supported values: 'completed', 'failed'.
        experiment_id: Optional explicit UUID. A new UUID is generated when
            omitted, which is the expected behaviour for all new runs.
        results_dir: Directory in which ``experiments.jsonl`` is written.

    Returns:
        Path to the JSONL file that was written to.

    Raises:
        ValueError: If required fields are empty, episodes ≤ 0, or any
            metric is NaN / infinite.
        TypeError: If *metrics* is not a dict or a metric value is bool.
    """
    if not algorithm.strip():
        raise ValueError("Algorithm must not be empty.")

    if not environment.strip():
        raise ValueError("Environment must not be empty.")

    if episodes <= 0:
        raise ValueError("Episodes must be greater than zero.")

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary.")

    _validate_metrics(metrics)

    valid_statuses = {"completed", "failed"}
    if status not in valid_statuses:
        raise ValueError(
            f"Status must be one of {sorted(valid_statuses)!r}; got {status!r}."
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id or str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "environment": environment,
        "seed": seed,
        "episodes": episodes,
        "status": status,
        "metrics": metrics,
        "model_path": model_path,
    }

    if hyperparameters is not None:
        record["hyperparameters"] = hyperparameters

    # Validate full JSON-serializability before touching the file.
    serialized = json.dumps(record, allow_nan=False)

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "experiments.jsonl"

    with output_file.open("a", encoding="utf-8") as file:
        file.write(serialized + "\n")

    return output_file
