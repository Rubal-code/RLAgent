
import json

import pytest

from rl_agent_lab.training import evaluate_multiple_seeds as evaluation


def test_calculate_metrics():
    metrics = evaluation.calculate_metrics([10, 20, 30, 40])

    assert metrics["average_reward"] == pytest.approx(25.0)
    assert metrics["final_100_average_reward"] == pytest.approx(25.0)
    assert metrics["best_reward"] == pytest.approx(40.0)


def test_calculate_metrics_rejects_empty_rewards():
    with pytest.raises(ValueError, match="Rewards cannot be empty"):
        evaluation.calculate_metrics([])


def test_summarize_runs_calculates_mean_and_std():
    runs = [
        {
            "algorithm": "DQN",
            "metrics": {
                "average_reward": 10.0,
                "final_100_average_reward": 20.0,
                "best_reward": 30.0,
            },
        },
        {
            "algorithm": "DQN",
            "metrics": {
                "average_reward": 20.0,
                "final_100_average_reward": 40.0,
                "best_reward": 50.0,
            },
        },
    ]

    summary = evaluation.summarize_runs(runs)

    assert summary["DQN"]["average_reward"]["mean"] == pytest.approx(15.0)
    assert summary["DQN"]["average_reward"]["std"] == pytest.approx(5.0)
    assert summary["DQN"]["final_100_average_reward"]["mean"] == pytest.approx(
        30.0
    )
    assert summary["DQN"]["best_reward"]["mean"] == pytest.approx(40.0)
    assert "REINFORCE" not in summary


def test_run_evaluation_saves_json(monkeypatch, tmp_path):
    def fake_dqn(episodes, seed):
        assert episodes == 3
        return object(), [10.0, 20.0, 30.0]

    def fake_reinforce(episodes, seed):
        assert episodes == 3
        return object(), [20.0, 30.0, 40.0]

    monkeypatch.setattr(evaluation, "train_dqn", fake_dqn)
    monkeypatch.setattr(evaluation, "train_reinforce", fake_reinforce)

    output_path = tmp_path / "evaluation.json"

    result = evaluation.run_evaluation(
        episodes=3,
        seeds=[42, 123],
        output_path=output_path,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(result["runs"]) == 4
    assert saved == result
    assert result["seeds"] == [42, 123]
    assert result["summary"]["DQN"]["average_reward"]["mean"] == pytest.approx(
        20.0
    )
    assert result["summary"]["REINFORCE"]["average_reward"]["mean"] == (
        30.0
    )


def test_run_evaluation_rejects_nonpositive_episodes(tmp_path):
    with pytest.raises(ValueError, match="greater than zero"):
        evaluation.run_evaluation(
            episodes=0,
            seeds=[42],
            output_path=tmp_path / "evaluation.json",
        )


def test_run_evaluation_rejects_empty_seeds(tmp_path):
    with pytest.raises(ValueError, match="at least one seed"):
        evaluation.run_evaluation(
            episodes=3,
            seeds=[],
            output_path=tmp_path / "evaluation.json",
        )


def test_run_evaluation_rejects_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        evaluation.run_evaluation(
            episodes=3,
            seeds=[42, 42],
            output_path=tmp_path / "evaluation.json",
        )


def test_run_evaluation_logs_each_run(monkeypatch, tmp_path):
    def fake_dqn(episodes, seed):
        return object(), [10.0, 20.0, 30.0]

    def fake_reinforce(episodes, seed):
        return object(), [20.0, 30.0, 40.0]

    monkeypatch.setattr(evaluation, "train_dqn", fake_dqn)
    monkeypatch.setattr(evaluation, "train_reinforce", fake_reinforce)

    output_path = tmp_path / "evaluation.json"

    evaluation.run_evaluation(
        episodes=3,
        seeds=[11, 22],
        output_path=output_path,
    )

    log_path = tmp_path / "experiments.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == 4
    assert {
        (record["algorithm"], record["seed"])
        for record in records
    } == {
        ("DQN", 11),
        ("DQN", 22),
        ("REINFORCE", 11),
        ("REINFORCE", 22),
    }
    assert all(record["episodes"] == 3 for record in records)
    assert all(record["environment"] == "CartPole-v1" for record in records)
    assert output_path.exists()

