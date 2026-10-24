
import json

import pytest

from rl_agent_lab.training.compare_experiments import (
    compare_experiments,
    load_experiments,
)


def test_load_experiments_reads_jsonl(tmp_path):
    path = tmp_path / "experiments.jsonl"
    records = [
        {"algorithm": "DQN", "metrics": {"average_reward": 100}},
        {"algorithm": "REINFORCE", "metrics": {"average_reward": 200}},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    assert load_experiments(path) == records


def test_load_experiments_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiments(tmp_path / "missing.jsonl")


def test_load_experiments_reports_invalid_json(tmp_path):
    path = tmp_path / "experiments.jsonl"
    path.write_text('{"algorithm": "DQN"}\ninvalid\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_experiments(path)


def test_compare_experiments_calculates_statistics():
    records = [
        {
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "metrics": {"average_reward": 100},
        },
        {
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "metrics": {"average_reward": 200},
        },
        {
            "algorithm": "REINFORCE",
            "environment": "CartPole-v1",
            "metrics": {"average_reward": 300},
        },
    ]

    result = compare_experiments(records)

    assert result["DQN"]["count"] == 2
    assert result["DQN"]["mean"] == 150
    assert result["DQN"]["std"] == 50
    assert result["REINFORCE"]["mean"] == 300
    assert result["REINFORCE"]["std"] == 0


def test_compare_experiments_filters_environment_and_algorithm():
    records = [
        {
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "metrics": {"average_reward": 100},
        },
        {
            "algorithm": "DQN",
            "environment": "LunarLander-v3",
            "metrics": {"average_reward": 500},
        },
        {
            "algorithm": "REINFORCE",
            "environment": "CartPole-v1",
            "metrics": {"average_reward": 200},
        },
    ]

    result = compare_experiments(
        records,
        environment="CartPole-v1",
        algorithms=["DQN"],
    )

    assert list(result) == ["DQN"]
    assert result["DQN"]["mean"] == 100


def test_compare_experiments_skips_missing_or_invalid_metrics():
    records = [
        {
            "algorithm": "DQN",
            "metrics": {"average_reward": 100},
        },
        {
            "algorithm": "DQN",
            "metrics": {},
        },
        {
            "algorithm": "DQN",
            "metrics": {"average_reward": float("nan")},
        },
        {
            "algorithm": "DQN",
            "metrics": {"average_reward": True},
        },
    ]

    result = compare_experiments(records)

    assert result["DQN"]["count"] == 1
    assert result["DQN"]["mean"] == 100


def test_compare_experiments_rejects_empty_metric():
    with pytest.raises(ValueError, match="Metric"):
        compare_experiments([], metric="  ")


def test_compare_experiments_returns_empty_for_no_matches():
    result = compare_experiments(
        [],
        environment="CartPole-v1",
    )

    assert result == {}
