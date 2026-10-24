"""FastAPI application for RLAgent experiment data.

All endpoints serve **read-only** data from the experiment tracking layer.
No endpoint writes to ``experiments.jsonl`` or modifies any file.

CORS note
---------
``CORSMiddleware`` is configured to allow all origins by default so that a
locally-running frontend (e.g. on port 5173) can call the API (on port 8000)
without CORS errors during development.  For production deployment restrict
``allow_origins`` to the actual frontend origin.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from rl_agent_lab.api import storage

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RLAgent API",
    version="1.0.0",
    description=(
        "Read-only REST API for RLAgent experiment tracking data. "
        "All results are sourced from the append-only JSONL experiment history."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production.
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Service health check",
    response_description="Service is running.",
    tags=["Health"],
)
def health() -> dict[str, str]:
    """Return a simple liveness response.

    Guaranteed to return HTTP 200 with ``{"status": "ok"}`` as long as the
    API process is running.  It does **not** verify that the JSONL file is
    readable; use ``/api/v1/dashboard/summary`` for a data-layer check.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

_PAGE_SIZE_DEFAULT = 20
_PAGE_SIZE_MAX = 200


@app.get(
    "/api/v1/experiments",
    summary="List experiments",
    tags=["Experiments"],
)
def list_experiments(
    algorithm: Annotated[
        str | None,
        Query(description="Filter by exact algorithm name."),
    ] = None,
    environment: Annotated[
        str | None,
        Query(description="Filter by exact environment name."),
    ] = None,
    status: Annotated[
        str | None,
        Query(description="Filter by status ('completed' or 'failed')."),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="1-based page number."),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=_PAGE_SIZE_MAX, description="Records per page (max 200)."),
    ] = _PAGE_SIZE_DEFAULT,
) -> dict[str, Any]:
    """Return a paginated list of experiment records.

    Filters (all optional, all exact-match):
    - ``algorithm`` — e.g. ``DQN``, ``REINFORCE``, ``PPO``, ``Q-Learning``
    - ``environment`` — e.g. ``CartPole-v1``, ``LunarLander-v3``
    - ``status`` — ``completed`` or ``failed``

    Response schema::

        {
          "page": 1,
          "page_size": 20,
          "total": 42,
          "total_pages": 3,
          "experiments": [ ... ]
        }

    Each experiment object contains all fields present in the JSONL record.
    Legacy records without ``schema_version`` or ``experiment_id`` are
    included as-is so historical data is never hidden.
    """
    records = storage.load_all()
    filtered = storage.filter_records(
        records,
        algorithm=algorithm,
        environment=environment,
        status=status,
    )
    page_records, total = storage.paginate(
        filtered, page=page, page_size=page_size
    )
    total_pages = max(1, -(-total // page_size))  # ceiling division

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "experiments": page_records,
    }


@app.get(
    "/api/v1/experiments/{experiment_id}",
    summary="Retrieve a single experiment",
    tags=["Experiments"],
)
def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Return the experiment record for *experiment_id*.

    ``experiment_id`` is the UUID assigned when the run was tracked.  Legacy
    records written before the UUID field was introduced will not be
    retrievable via this endpoint; use ``GET /api/v1/experiments`` to list
    them.

    Raises HTTP 404 when no matching record is found.
    """
    records = storage.load_all()
    record = storage.get_by_id(records, experiment_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return record


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/dashboard/summary",
    summary="Aggregated experiment summary",
    tags=["Dashboard"],
)
def dashboard_summary() -> dict[str, Any]:
    """Return high-level statistics across all tracked experiments.

    Response schema::

        {
          "total_experiments": 42,
          "algorithms": ["DQN", "PPO", "Q-Learning", "REINFORCE"],
          "environments": ["CartPole-v1", "GridWorld", "LunarLander-v3"],
          "status_counts": {"completed": 40, "failed": 2},
          "metric_averages": {
            "average_reward": 123.45,
            "final_100_average_reward": 200.0,
            "best_reward": 450.0
          }
        }

    ``metric_averages`` is computed over **all** records carrying that metric
    and excludes NaN / infinite values.  Averages for different
    algorithm–environment combinations are therefore not directly comparable;
    use ``/api/v1/dashboard/algorithms`` for algorithm-level breakdowns.
    """
    records = storage.load_all()
    return storage.build_summary(records)


@app.get(
    "/api/v1/dashboard/algorithms",
    summary="Algorithm-level metric comparison",
    tags=["Dashboard"],
)
def dashboard_algorithms(
    metric: Annotated[
        str,
        Query(description="Metric name to compare (default: average_reward)."),
    ] = "average_reward",
    environment: Annotated[
        str | None,
        Query(description="Restrict comparison to one environment."),
    ] = None,
) -> dict[str, Any]:
    """Return per-algorithm statistics for a chosen metric.

    ``std`` is the **population** standard deviation (denominator *N*), which
    is consistent with the rest of the project.  For single-run algorithms the
    std will be 0.

    Response schema::

        {
          "metric": "average_reward",
          "environment": null,
          "algorithms": [
            {
              "algorithm": "DQN",
              "count": 5,
              "mean": 196.18,
              "std": 42.3,
              "environment": null
            },
            ...
          ]
        }
    """
    if not metric.strip():
        raise HTTPException(
            status_code=422, detail="'metric' must not be blank."
        )

    records = storage.load_all()
    comparison = storage.build_algorithm_comparison(
        records,
        metric=metric,
        environment=environment,
    )

    return {
        "metric": metric,
        "environment": environment,
        "algorithms": comparison,
    }


@app.get(
    "/api/v1/dashboard/rewards",
    summary="Time-ordered reward / metric series",
    tags=["Dashboard"],
)
def dashboard_rewards(
    metric: Annotated[
        str,
        Query(description="Metric name (default: average_reward)."),
    ] = "average_reward",
    algorithm: Annotated[
        str | None,
        Query(description="Filter by algorithm."),
    ] = None,
    environment: Annotated[
        str | None,
        Query(description="Filter by environment."),
    ] = None,
) -> dict[str, Any]:
    """Return a time-ordered series of metric values for charting.

    Each data point corresponds to one tracked experiment run.  Points are
    in append order (oldest first).

    Response schema::

        {
          "metric": "average_reward",
          "algorithm": "DQN",
          "environment": "CartPole-v1",
          "series": [
            {
              "experiment_id": "uuid-or-null",
              "timestamp_utc": "2026-10-22T09:25:21+00:00",
              "algorithm": "DQN",
              "environment": "CartPole-v1",
              "seed": 42,
              "episodes": 500,
              "value": 196.184
            },
            ...
          ]
        }
    """
    if not metric.strip():
        raise HTTPException(
            status_code=422, detail="'metric' must not be blank."
        )

    records = storage.load_all()
    series = storage.build_reward_series(
        records,
        metric=metric,
        algorithm=algorithm,
        environment=environment,
    )

    return {
        "metric": metric,
        "algorithm": algorithm,
        "environment": environment,
        "series": series,
    }


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def start() -> None:
    """Start the API server (registered as the ``rl-api`` console script)."""
    import uvicorn

    uvicorn.run(
        "rl_agent_lab.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    start()
