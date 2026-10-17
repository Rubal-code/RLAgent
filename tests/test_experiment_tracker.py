
import json

import pytest

from rl_agent_lab.training.experiment_tracker import log_experiment


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


def test_log_experiment_rejects_non_json_metrics(tmp_path):
    with pytest.raises(ValueError):
        log_experiment(
            algorithm="DQN",
            environment="CartPole-v1",
            seed=42,
            episodes=100,
            metrics={"average_reward": float("nan")},
            results_dir=tmp_path,
        )
