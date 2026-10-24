"""API contract tests.

These tests use FastAPI's ``TestClient`` (backed by ``httpx``) to make real
HTTP requests against the application and verify that response structures
remain stable.  The JSONL data path is redirected to a temporary file so
production data is never touched.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_records() -> list[dict]:
    """Three representative experiment records."""
    return [
        {
            "schema_version": 1,
            "experiment_id": str(uuid.uuid4()),
            "timestamp_utc": "2026-10-22T09:25:21+00:00",
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "seed": 42,
            "episodes": 500,
            "status": "completed",
            "metrics": {
                "average_reward": 196.184,
                "final_100_average_reward": 339.26,
                "best_reward": 500.0,
            },
            "model_path": "models/cartpole_dqn.pth",
            "hyperparameters": {"learning_rate": 0.001, "discount_factor": 0.99},
        },
        {
            "schema_version": 1,
            "experiment_id": str(uuid.uuid4()),
            "timestamp_utc": "2026-10-22T09:33:51+00:00",
            "algorithm": "REINFORCE",
            "environment": "CartPole-v1",
            "seed": 42,
            "episodes": 500,
            "status": "completed",
            "metrics": {
                "average_reward": 267.898,
                "final_100_average_reward": 470.34,
                "best_reward": 500.0,
            },
            "model_path": "models/cartpole_reinforce.pth",
        },
        {
            "schema_version": 1,
            "experiment_id": str(uuid.uuid4()),
            "timestamp_utc": "2026-10-23T17:28:57+00:00",
            "algorithm": "PPO",
            "environment": "LunarLander-v3",
            "seed": 42,
            "episodes": 100,
            "status": "completed",
            "metrics": {
                "average_reward": -193.8,
                "final_100_average_reward": -193.8,
                "best_reward": 35.57,
            },
            "model_path": "models/lunarlander_ppo.pth",
        },
    ]


@pytest.fixture()
def experiments_jsonl(tmp_path: Path, sample_records: list[dict]) -> Path:
    """Write sample records to a temp JSONL and return its path."""
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in sample_records) + "\n",
        encoding="utf-8",
    )
    return jsonl


@pytest.fixture()
def client(experiments_jsonl: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with RL_EXPERIMENTS_FILE pointing at the temp JSONL."""
    monkeypatch.setenv("RL_EXPERIMENTS_FILE", str(experiments_jsonl))
    # Reload storage so the env var is picked up.
    import importlib

    from rl_agent_lab.api import storage

    importlib.reload(storage)
    from rl_agent_lab.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as tc:
        yield tc


@pytest.fixture()
def empty_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient backed by an empty JSONL file."""
    empty_jsonl = tmp_path / "experiments.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    monkeypatch.setenv("RL_EXPERIMENTS_FILE", str(empty_jsonl))

    import importlib

    from rl_agent_lab.api import storage

    importlib.reload(storage)
    from rl_agent_lab.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as tc:
        yield tc


@pytest.fixture()
def missing_file_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient where the JSONL file does not exist."""
    monkeypatch.setenv(
        "RL_EXPERIMENTS_FILE", str(tmp_path / "nonexistent.jsonl")
    )

    import importlib

    from rl_agent_lab.api import storage

    importlib.reload(storage)
    from rl_agent_lab.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/v1/experiments
# ---------------------------------------------------------------------------


def test_list_experiments_returns_all_records(client, sample_records):
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["experiments"]) == 3


def test_list_experiments_response_schema(client):
    resp = client.get("/api/v1/experiments")
    body = resp.json()
    for key in ("page", "page_size", "total", "total_pages", "experiments"):
        assert key in body, f"Missing key: {key}"


def test_list_experiments_filter_by_algorithm(client):
    resp = client.get("/api/v1/experiments?algorithm=DQN")
    body = resp.json()
    assert body["total"] == 1
    assert body["experiments"][0]["algorithm"] == "DQN"


def test_list_experiments_filter_by_environment(client):
    resp = client.get("/api/v1/experiments?environment=LunarLander-v3")
    body = resp.json()
    assert body["total"] == 1
    assert body["experiments"][0]["algorithm"] == "PPO"


def test_list_experiments_filter_by_status(client):
    resp = client.get("/api/v1/experiments?status=completed")
    body = resp.json()
    assert body["total"] == 3


def test_list_experiments_pagination(client, sample_records):
    resp = client.get("/api/v1/experiments?page=1&page_size=2")
    body = resp.json()
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert body["total"] == 3
    assert body["total_pages"] == 2
    assert len(body["experiments"]) == 2

    resp2 = client.get("/api/v1/experiments?page=2&page_size=2")
    body2 = resp2.json()
    assert len(body2["experiments"]) == 1


def test_list_experiments_empty_result(empty_client):
    resp = empty_client.get("/api/v1/experiments")
    body = resp.json()
    assert resp.status_code == 200
    assert body["total"] == 0
    assert body["experiments"] == []


def test_list_experiments_missing_file(missing_file_client):
    resp = missing_file_client.get("/api/v1/experiments")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_list_experiments_invalid_page_size_returns_422(client):
    resp = client.get("/api/v1/experiments?page_size=0")
    assert resp.status_code == 422


def test_list_experiments_page_size_too_large_returns_422(client):
    resp = client.get("/api/v1/experiments?page_size=201")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/experiments/{experiment_id}
# ---------------------------------------------------------------------------


def test_get_experiment_returns_record(client, sample_records):
    target_id = sample_records[0]["experiment_id"]
    resp = client.get(f"/api/v1/experiments/{target_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment_id"] == target_id
    assert body["algorithm"] == "DQN"


def test_get_experiment_not_found_returns_404(client):
    resp = client.get(f"/api/v1/experiments/{uuid.uuid4()}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/summary
# ---------------------------------------------------------------------------


def test_dashboard_summary_schema(client):
    resp = client.get("/api/v1/dashboard/summary")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "total_experiments",
        "algorithms",
        "environments",
        "status_counts",
        "metric_averages",
    ):
        assert key in body, f"Missing key: {key}"


def test_dashboard_summary_values(client, sample_records):
    resp = client.get("/api/v1/dashboard/summary")
    body = resp.json()
    assert body["total_experiments"] == 3
    assert "DQN" in body["algorithms"]
    assert "REINFORCE" in body["algorithms"]
    assert "PPO" in body["algorithms"]
    assert "CartPole-v1" in body["environments"]
    assert "LunarLander-v3" in body["environments"]
    assert body["status_counts"].get("completed") == 3
    assert "average_reward" in body["metric_averages"]


def test_dashboard_summary_empty(empty_client):
    resp = empty_client.get("/api/v1/dashboard/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_experiments"] == 0
    assert body["algorithms"] == []
    assert body["metric_averages"] == {}


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/algorithms
# ---------------------------------------------------------------------------


def test_dashboard_algorithms_schema(client):
    resp = client.get("/api/v1/dashboard/algorithms")
    assert resp.status_code == 200
    body = resp.json()
    assert "metric" in body
    assert "algorithms" in body
    for entry in body["algorithms"]:
        for key in ("algorithm", "count", "mean", "std"):
            assert key in entry


def test_dashboard_algorithms_cartpole_filter(client):
    resp = client.get(
        "/api/v1/dashboard/algorithms?environment=CartPole-v1"
    )
    body = resp.json()
    alg_names = [a["algorithm"] for a in body["algorithms"]]
    assert "DQN" in alg_names
    assert "REINFORCE" in alg_names
    assert "PPO" not in alg_names


def test_dashboard_algorithms_blank_metric_returns_422(client):
    resp = client.get("/api/v1/dashboard/algorithms?metric= ")
    assert resp.status_code == 422


def test_dashboard_algorithms_empty_data(empty_client):
    resp = empty_client.get("/api/v1/dashboard/algorithms")
    assert resp.status_code == 200
    assert resp.json()["algorithms"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/rewards
# ---------------------------------------------------------------------------


def test_dashboard_rewards_schema(client):
    resp = client.get("/api/v1/dashboard/rewards")
    assert resp.status_code == 200
    body = resp.json()
    assert "metric" in body
    assert "series" in body
    for point in body["series"]:
        for key in ("timestamp_utc", "algorithm", "environment", "value"):
            assert key in point


def test_dashboard_rewards_filter_algorithm(client):
    resp = client.get("/api/v1/dashboard/rewards?algorithm=DQN")
    body = resp.json()
    assert len(body["series"]) == 1
    assert body["series"][0]["algorithm"] == "DQN"


def test_dashboard_rewards_blank_metric_returns_422(client):
    resp = client.get("/api/v1/dashboard/rewards?metric=  ")
    assert resp.status_code == 422


def test_dashboard_rewards_empty_data(empty_client):
    resp = empty_client.get("/api/v1/dashboard/rewards")
    assert resp.status_code == 200
    assert resp.json()["series"] == []


# ---------------------------------------------------------------------------
# Malformed JSONL tolerance
# ---------------------------------------------------------------------------


def test_list_experiments_tolerates_malformed_lines(
    tmp_path, monkeypatch, sample_records
):
    """A corrupt line in the middle must not crash valid-record queries."""
    jsonl = tmp_path / "experiments.jsonl"
    lines = [json.dumps(sample_records[0]), "NOT_JSON", json.dumps(sample_records[1])]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setenv("RL_EXPERIMENTS_FILE", str(jsonl))

    import importlib

    from rl_agent_lab.api import storage

    importlib.reload(storage)
    from rl_agent_lab.api import main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as tc:
        resp = tc.get("/api/v1/experiments")
        body = resp.json()
        assert resp.status_code == 200
        assert body["total"] == 2  # corrupt line skipped
