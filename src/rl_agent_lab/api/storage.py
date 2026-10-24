"""Storage layer: read experiment records from the JSONL history file.

All functions in this module are **read-only**. They never modify
``experiments.jsonl`` or any other file.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default path — overridable via environment variable for tests / deployment.
# ---------------------------------------------------------------------------

_DEFAULT_EXPERIMENTS_FILE = Path("results") / "experiments.jsonl"


def _experiments_path() -> Path:
    """Return the configured experiment JSONL path.

    The environment variable ``RL_EXPERIMENTS_FILE`` overrides the default
    ``results/experiments.jsonl`` (resolved relative to the current working
    directory at runtime).
    """
    env = os.environ.get("RL_EXPERIMENTS_FILE")
    if env:
        return Path(env)
    return _DEFAULT_EXPERIMENTS_FILE


def _is_valid_metric(value: Any) -> bool:
    """Return True when *value* is a finite numeric scalar."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def load_all(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Load every well-formed record from the JSONL file.

    Malformed lines (invalid JSON or non-object values) are **skipped** with
    a ``continue`` rather than raising, so that one corrupt record never
    blocks access to valid historical data.  This is intentional: the API
    must degrade gracefully on a partially-corrupt file.

    Args:
        path: Override the JSONL file path.  Useful for tests.

    Returns:
        List of record dicts, in append order (oldest first).
    """
    target = path or _experiments_path()

    if not target.is_file():
        return []

    records: list[dict[str, Any]] = []

    with target.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            records.append(record)

    return records


def filter_records(
    records: list[dict[str, Any]],
    *,
    algorithm: str | None = None,
    environment: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return the subset of *records* matching optional filter criteria.

    All filters use exact-string equality.  A ``None`` filter accepts any
    value (including missing keys).
    """
    out = []
    for rec in records:
        if algorithm is not None and rec.get("algorithm") != algorithm:
            continue
        if environment is not None and rec.get("environment") != environment:
            continue
        if status is not None and rec.get("status") != status:
            continue
        out.append(rec)
    return out


def get_by_id(
    records: list[dict[str, Any]],
    experiment_id: str,
) -> dict[str, Any] | None:
    """Return the first record matching *experiment_id*, or ``None``."""
    for rec in records:
        if rec.get("experiment_id") == experiment_id:
            return rec
    return None


def paginate(
    records: list[dict[str, Any]],
    *,
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Slice *records* for the requested page.

    Args:
        records: Full (filtered) list of records, oldest first.
        page: 1-based page number.
        page_size: Maximum number of records per page.

    Returns:
        Tuple of ``(page_records, total_count)``.  *page_records* is empty
        when *page* is beyond the last page.
    """
    total = len(records)
    start = (page - 1) * page_size
    end = start + page_size
    return records[start:end], total


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dashboard summary statistics from *records*.

    Returns a dict with:
    - ``total_experiments``: total record count.
    - ``algorithms``: sorted list of distinct algorithm names.
    - ``environments``: sorted list of distinct environment names.
    - ``status_counts``: mapping of status → count.
    - ``metric_averages``: average of each numeric metric across all records
      that carry it (NaN/inf excluded).
    """
    algorithms: set[str] = set()
    environments: set[str] = set()
    status_counts: dict[str, int] = {}
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    for rec in records:
        alg = rec.get("algorithm")
        if isinstance(alg, str) and alg.strip():
            algorithms.add(alg)

        env = rec.get("environment")
        if isinstance(env, str) and env.strip():
            environments.add(env)

        st = rec.get("status", "completed")
        if isinstance(st, str):
            status_counts[st] = status_counts.get(st, 0) + 1

        metrics = rec.get("metrics")
        if isinstance(metrics, dict):
            for key, val in metrics.items():
                if _is_valid_metric(val):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(val)
                    metric_counts[key] = metric_counts.get(key, 0) + 1

    metric_averages = {
        key: metric_sums[key] / metric_counts[key]
        for key in metric_sums
    }

    return {
        "total_experiments": len(records),
        "algorithms": sorted(algorithms),
        "environments": sorted(environments),
        "status_counts": status_counts,
        "metric_averages": metric_averages,
    }


def build_algorithm_comparison(
    records: list[dict[str, Any]],
    *,
    metric: str = "average_reward",
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-algorithm statistics for a given metric.

    Skips records where the metric is missing, non-numeric, NaN, or infinite.
    Uses **population** standard deviation (matches the rest of the project).

    Args:
        records: All experiment records (pre-filtered if desired).
        metric: Name of the metric to compare.
        environment: If given, only include records for this environment.

    Returns:
        List of dicts sorted by algorithm name, each containing:
        ``algorithm``, ``count``, ``mean``, ``std`` (population std),
        ``environment`` (or ``null`` when multiple envs are present).
    """
    import statistics

    grouped: dict[str, list[float]] = {}

    for rec in records:
        if environment is not None and rec.get("environment") != environment:
            continue

        alg = rec.get("algorithm")
        if not isinstance(alg, str) or not alg.strip():
            continue

        metrics = rec.get("metrics")
        if not isinstance(metrics, dict):
            continue

        val = metrics.get(metric)
        if not _is_valid_metric(val):
            continue

        grouped.setdefault(alg, []).append(float(val))  # type: ignore[arg-type]

    result = []
    for alg in sorted(grouped):
        values = grouped[alg]
        result.append(
            {
                "algorithm": alg,
                "count": len(values),
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "environment": environment,
            }
        )
    return result


def build_reward_series(
    records: list[dict[str, Any]],
    *,
    metric: str = "average_reward",
    algorithm: str | None = None,
    environment: str | None = None,
) -> list[dict[str, Any]]:
    """Return a time-ordered series of metric values for charting.

    Each point in the series is a dict with:
    ``experiment_id``, ``timestamp_utc``, ``algorithm``, ``environment``,
    ``seed``, ``episodes``, ``value``.

    Records where the metric is missing or non-finite are excluded.
    """
    series = []

    for rec in records:
        if algorithm is not None and rec.get("algorithm") != algorithm:
            continue
        if environment is not None and rec.get("environment") != environment:
            continue

        metrics = rec.get("metrics")
        if not isinstance(metrics, dict):
            continue

        val = metrics.get(metric)
        if not _is_valid_metric(val):
            continue

        series.append(
            {
                "experiment_id": rec.get("experiment_id"),
                "timestamp_utc": rec.get("timestamp_utc"),
                "algorithm": rec.get("algorithm"),
                "environment": rec.get("environment"),
                "seed": rec.get("seed"),
                "episodes": rec.get("episodes"),
                "value": float(val),  # type: ignore[arg-type]
            }
        )

    return series
