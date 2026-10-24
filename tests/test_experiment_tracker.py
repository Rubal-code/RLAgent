
import json
import uuid

import pytest

from rl_agent_lab.training.experiment_tracker import (
    SCHEMA_VERSION,
    log_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_record(tmp_path, **overrides):
    """Log one experiment and return the parsed record dict."""
    kwargs = dict(
        algorithm="DQN",
        environment="CartPole-v1",
        seed=42,
        episodes=500,
        metrics={"average_reward": 450.5},
        results_dir=tmp_path,
    )
    kwargs.update(overrides)
    output_file = log_experiment(**kwargs)
    return json.loads(output_file.read_text(encoding="utf-8").strip())


# ---------------------------------------------------------------------------
# Basic write and field checks
# ---------------------------------------------------------------------------


def test_log_experiment_writes_record(tmp_path):
    output_file = log_experiment(
        algorithm="DQN",
        environment="CartPole-v1",
        seed=42,
        episodes=500,
        metrics={"average_reward": 450.5, "success_rate": 0.9},
        model_path="models/cartpole_dqn.pth",
        results_dir=tmp_path,
    )

    assert output_file.exists()

    record = json.loads(output_file.read_text(encoding="utf-8"))

    assert record["algorithm"] == "DQN"
    assert record["environment"] == "CartPole-v1"
    assert record["seed"] == 42
    assert record["episodes"] == 500
    assert record["metrics"]["average_reward"] == 450.5
    assert record["model_path"] == "models/cartpole_dqn.pth"
    assert record["timestamp_utc"]


def test_log_experiment_includes_schema_version(tmp_path):
    record = _single_record(tmp_path)
    assert record["schema_version"] == SCHEMA_VERSION


def test_log_experiment_generates_uuid(tmp_path):
    record = _single_record(tmp_path)
    eid = record.get("experiment_id")
    assert eid is not None
    # Must be a valid UUID string.
    uuid.UUID(eid)


def test_log_experiment_accepts_explicit_experiment_id(tmp_path):
    explicit_id = str(uuid.uuid4())
    record = _single_record(tmp_path, experiment_id=explicit_id)
    assert record["experiment_id"] == explicit_id


def test_log_experiment_includes_default_status(tmp_path):
    record = _single_record(tmp_path)
    assert record["status"] == "completed"


def test_log_experiment_accepts_failed_status(tmp_path):
    record = _single_record(tmp_path, status="failed")
    assert record["status"] == "failed"


def test_log_experiment_stores_hyperparameters(tmp_path):
    hparams = {"learning_rate": 0.001, "discount_factor": 0.99}
    record = _single_record(tmp_path, hyperparameters=hparams)
    assert record["hyperparameters"] == hparams


def test_log_experiment_omits_hyperparameters_when_none(tmp_path):
    record = _single_record(tmp_path)
    # hyperparameters key should not appear when not supplied.
    assert "hyperparameters" not in record


# ---------------------------------------------------------------------------
# Append behaviour
# ---------------------------------------------------------------------------


def test_log_experiment_appends_records(tmp_path):
    for seed in (1, 2):
        log_experiment(
            algorithm="Q-Learning",
            environment="GridWorld",
            seed=seed,
            episodes=100,
            metrics={"success_rate": 0.8},
            results_dir=tmp_path,
        )

    output_file = tmp_path / "experiments.jsonl"
    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 2
    assert [record["seed"] for record in records] == [1, 2]


def test_log_experiment_generates_unique_ids_per_call(tmp_path):
    for _ in range(3):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={"average_reward": 100.0},
            results_dir=tmp_path,
        )

    output_file = tmp_path / "experiments.jsonl"
    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    ids = [r["experiment_id"] for r in records]
    assert len(set(ids)) == 3, "Each call must produce a distinct experiment_id"


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_log_experiment_rejects_empty_algorithm(tmp_path):
    with pytest.raises(ValueError, match="Algorithm"):
        log_experiment(
            algorithm="  ",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={},
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_empty_environment(tmp_path):
    with pytest.raises(ValueError, match="Environment"):
        log_experiment(
            algorithm="DQN",
            environment="",
            seed=42,
            episodes=100,
            metrics={},
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_invalid_episode_count(tmp_path):
    with pytest.raises(ValueError, match="Episodes"):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=0,
            metrics={},
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_invalid_status(tmp_path):
    with pytest.raises(ValueError, match="Status"):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={},
            status="unknown",
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_non_dict_metrics(tmp_path):
    with pytest.raises(TypeError, match="Metrics"):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics="not-a-dict",  # type: ignore[arg-type]
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_nan_metric(tmp_path):
    with pytest.raises((ValueError, TypeError)):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={"average_reward": float("nan")},
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_inf_metric(tmp_path):
    with pytest.raises((ValueError, TypeError)):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={"average_reward": float("inf")},
            results_dir=tmp_path,
        )


def test_log_experiment_rejects_bool_metric(tmp_path):
    with pytest.raises(TypeError, match="bool"):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={"flag": True},
            results_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_load_legacy_record_without_schema_version(tmp_path):
    """Records written by the old tracker (no schema_version) can be read."""
    legacy = {
        "timestamp_utc": "2026-10-17T09:28:56.618751+00:00",
        "algorithm": "Q-Learning",
        "environment": "GridWorld",
        "seed": 42,
        "episodes": 100,
        "metrics": {"average_reward": -91.23},
        "model_path": "models/q_table.json",
    }
    jsonl_path = tmp_path / "experiments.jsonl"
    jsonl_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    # compare_experiments should read it without error.
    from rl_agent_lab.training.compare_experiments import load_experiments

    records = load_experiments(jsonl_path)
    assert len(records) == 1
    assert records[0]["algorithm"] == "Q-Learning"
