
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_experiment(
    *,
    algorithm: str,
    environment: str,
    seed: int,
    episodes: int,
    metrics: dict[str, Any],
    model_path: str | None = None,
    results_dir: str | Path = "results",
) -> Path:
    """Append experiment metadata and metrics to a JSON Lines file."""

    if not algorithm.strip():
        raise ValueError("Algorithm must not be empty.")

    if not environment.strip():
        raise ValueError("Environment must not be empty.")

    if episodes <= 0:
        raise ValueError("Episodes must be greater than zero.")

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary.")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "environment": environment,
        "seed": seed,
        "episodes": episodes,
        "metrics": metrics,
        "model_path": model_path,
    }

    # Validate JSON compatibility before writing anything.
    serialized = json.dumps(record, allow_nan=False)

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "experiments.jsonl"

    with output_file.open("a", encoding="utf-8") as file:
        file.write(serialized + "\n")

    return output_file
