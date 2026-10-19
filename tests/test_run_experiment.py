
import random
from pathlib import Path

import pytest

from rl_agent_lab.training import run_experiment


class DummyAgent:
    """Minimal stand-in for a trained agent."""


def test_runner_saves_model_and_logs_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train",
        lambda episodes: (DummyAgent(), [-10.0, 5.0], [False, True]),
    )

    def fake_save_q_table(agent, path):
        Path(path).write_text("saved-model", encoding="utf-8")

    recorded = {}

    def fake_log_experiment(**kwargs):
        recorded.update(kwargs)
        return tmp_path / "results" / "experiments.jsonl"

    monkeypatch.setattr(run_experiment, "save_q_table", fake_save_q_table)
    monkeypatch.setattr(run_experiment, "log_experiment", fake_log_experiment)

    run_experiment.run_q_learning(episodes=2, seed=42)

    assert (tmp_path / "models" / "q_table.json").read_text(
        encoding="utf-8"
    ) == "saved-model"

    assert recorded["algorithm"] == "Q-Learning"
    assert recorded["environment"] == "GridWorld"
    assert recorded["seed"] == 42
    assert recorded["episodes"] == 2
    assert recorded["metrics"]["average_reward"] == pytest.approx(-2.5)
    assert recorded["metrics"]["success_rate"] == pytest.approx(0.5)


def test_runner_reproduces_random_sequence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_train(episodes):
        return DummyAgent(), [random.random()], [True]

    monkeypatch.setattr(run_experiment, "train", fake_train)
    monkeypatch.setattr(
        run_experiment,
        "save_q_table",
        lambda agent, path: Path(path).write_text("model", encoding="utf-8"),
    )

    recorded_metrics = []

    def fake_log_experiment(**kwargs):
        recorded_metrics.append(kwargs["metrics"])
        return tmp_path / "experiments.jsonl"

    monkeypatch.setattr(run_experiment, "log_experiment", fake_log_experiment)

    run_experiment.run_q_learning(episodes=1, seed=123)
    run_experiment.run_q_learning(episodes=1, seed=123)

    assert recorded_metrics[0] == recorded_metrics[1]


def test_runner_rejects_empty_training_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train",
        lambda episodes: (DummyAgent(), [], []),
    )

    with pytest.raises(RuntimeError, match="no results"):
        run_experiment.run_q_learning(episodes=1, seed=42)
