
from pathlib import Path

import pytest

from rl_agent_lab.training import run_experiment

# -------------------- Dummy agents --------------------


class DummyAgent:
    pass


class DummyPolicy:
    def state_dict(self):
        return {"weights": [1.0, 2.0, 3.0]}


class DummyPPOAgent:
    policy = DummyPolicy()


# -------------------- Q-Learning tests --------------------


def test_runner_saves_model_and_logs_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_train(episodes):
        assert episodes == 3
        return DummyAgent(), [1.0, 2.0, 3.0], [True, False, True]

    def fake_save_q_table(agent, path):
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("saved-model", encoding="utf-8")

    recorded = {}

    def fake_log_experiment(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(
        run_experiment, "train_q_learning", fake_train
    )
    monkeypatch.setattr(
        run_experiment, "save_q_table", fake_save_q_table
    )
    monkeypatch.setattr(
        run_experiment, "log_experiment", fake_log_experiment
    )

    run_experiment.run_q_learning(episodes=3, seed=42)

    assert (
        Path("models/q_table.json").read_text(encoding="utf-8")
        == "saved-model"
    )
    assert recorded["algorithm"] == "Q-Learning"
    assert recorded["environment"] == "GridWorld"
    assert recorded["episodes"] == 3
    assert recorded["seed"] == 42
    assert recorded["metrics"]["average_reward"] == pytest.approx(2.0)
    assert recorded["metrics"]["success_rate"] == pytest.approx(2 / 3)
    assert recorded["metrics"]["final_100_average_reward"] == pytest.approx(
        2.0
    )


def test_runner_reproduces_random_sequence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    import random

    observed = []

    def fake_train(episodes):
        observed.append(random.random())
        return DummyAgent(), [1.0], [True]

    monkeypatch.setattr(
        run_experiment, "train_q_learning", fake_train
    )
    monkeypatch.setattr(
        run_experiment, "save_q_table", lambda agent, path: None
    )
    monkeypatch.setattr(
        run_experiment, "log_experiment", lambda **kwargs: None
    )

    run_experiment.run_q_learning(episodes=1, seed=123)
    run_experiment.run_q_learning(episodes=1, seed=123)

    assert observed[0] == observed[1]


def test_runner_rejects_empty_training_results(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train_q_learning",
        lambda episodes: (DummyAgent(), [], []),
    )

    with pytest.raises(RuntimeError, match="empty results"):
        run_experiment.run_q_learning(episodes=1, seed=42)


# -------------------- DQN tests --------------------


def test_dqn_runner_saves_model_and_logs_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train_dqn",
        lambda episodes, seed: (DummyAgent(), [10.0, 20.0, 30.0]),
    )

    def fake_save_model(agent, path):
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("saved-dqn-model", encoding="utf-8")

    recorded = {}

    def fake_log_experiment(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(run_experiment, "save_model", fake_save_model)
    monkeypatch.setattr(
        run_experiment, "log_experiment", fake_log_experiment
    )

    run_experiment.run_dqn(episodes=3, seed=7)

    assert (
        Path("models/cartpole_dqn.pth").read_text(encoding="utf-8")
        == "saved-dqn-model"
    )
    assert recorded["algorithm"] == "DQN"
    assert recorded["environment"] == "CartPole-v1"
    assert recorded["episodes"] == 3
    assert recorded["seed"] == 7
    assert recorded["metrics"]["average_reward"] == pytest.approx(20.0)
    assert recorded["metrics"]["final_100_average_reward"] == pytest.approx(
        20.0
    )
    assert recorded["metrics"]["best_reward"] == pytest.approx(30.0)


def test_dqn_runner_rejects_empty_training_results(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "train_dqn",
        lambda episodes, seed: (DummyAgent(), []),
    )

    with pytest.raises(RuntimeError, match="empty results"):
        run_experiment.run_dqn(episodes=1, seed=42)


# -------------------- REINFORCE tests --------------------


def test_reinforce_runner_saves_model_and_logs_metrics(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train_reinforce",
        lambda episodes, seed: (DummyAgent(), [10.0, 20.0, 30.0]),
    )

    def fake_save_model(agent, path):
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("saved-reinforce-model", encoding="utf-8")

    recorded = {}

    def fake_log_experiment(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(
        run_experiment, "save_reinforce_model", fake_save_model
    )
    monkeypatch.setattr(
        run_experiment, "log_experiment", fake_log_experiment
    )

    run_experiment.run_reinforce(episodes=3, seed=12)

    assert (
        Path("models/cartpole_reinforce.pth").read_text(
            encoding="utf-8"
        )
        == "saved-reinforce-model"
    )
    assert recorded["algorithm"] == "REINFORCE"
    assert recorded["environment"] == "CartPole-v1"
    assert recorded["episodes"] == 3
    assert recorded["seed"] == 12
    assert recorded["metrics"]["average_reward"] == pytest.approx(20.0)
    assert recorded["metrics"]["final_100_average_reward"] == pytest.approx(
        20.0
    )
    assert recorded["metrics"]["best_reward"] == pytest.approx(30.0)


def test_reinforce_runner_rejects_empty_training_results(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "train_reinforce",
        lambda episodes, seed: (DummyAgent(), []),
    )

    with pytest.raises(RuntimeError, match="empty results"):
        run_experiment.run_reinforce(episodes=1, seed=42)


# -------------------- PPO tests --------------------


def test_ppo_runner_saves_model_and_logs_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_train_ppo(episodes, seed, resume):
        assert episodes == 3
        assert seed == 12
        assert resume is False
        return DummyPPOAgent(), [10.0, 20.0, 30.0]

    def fake_torch_save(state_dict, path):
        assert state_dict == {"weights": [1.0, 2.0, 3.0]}
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("saved-ppo-model", encoding="utf-8")

    recorded = {}

    def fake_log_experiment(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(run_experiment, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(run_experiment.torch, "save", fake_torch_save)
    monkeypatch.setattr(
        run_experiment, "log_experiment", fake_log_experiment
    )

    run_experiment.run_ppo(episodes=3, seed=12)

    assert (
        Path("models/lunarlander_ppo.pth").read_text(encoding="utf-8")
        == "saved-ppo-model"
    )
    assert recorded["algorithm"] == "PPO"
    assert recorded["environment"] == "LunarLander-v3"
    assert recorded["episodes"] == 3
    assert recorded["seed"] == 12
    assert recorded["metrics"]["average_reward"] == pytest.approx(20.0)
    assert recorded["metrics"]["final_100_average_reward"] == pytest.approx(
        20.0
    )
    assert recorded["metrics"]["best_reward"] == pytest.approx(30.0)


def test_ppo_runner_rejects_empty_training_results(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "train_ppo",
        lambda episodes, seed, resume: (DummyPPOAgent(), []),
    )

    with pytest.raises(RuntimeError, match="empty results"):
        run_experiment.run_ppo(episodes=1, seed=42)


# -------------------- Tic-Tac-Toe tests --------------------


def test_tictactoe_runner_trains_and_evaluates(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    model_path = Path("models/tictactoe_q_table.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    calls = {}

    def fake_train_tictactoe(episodes, seed):
        assert episodes == 3
        assert seed == 12
        model_path.write_bytes(b"dummy-model")

    def fake_evaluate_tictactoe(episodes, seed):
        calls["evaluation_episodes"] = episodes
        calls["evaluation_seed"] = seed

    monkeypatch.setattr(
        run_experiment, "train_tictactoe", fake_train_tictactoe
    )
    monkeypatch.setattr(
        run_experiment, "evaluate_tictactoe", fake_evaluate_tictactoe
    )

    run_experiment.run_tictactoe(episodes=3, seed=12)

    assert model_path.exists()
    assert model_path.read_bytes() == b"dummy-model"
    assert calls["evaluation_episodes"] == 1000
    assert calls["evaluation_seed"] == 12


def test_tictactoe_runner_rejects_missing_model(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        run_experiment,
        "train_tictactoe",
        lambda episodes, seed: None,
    )

    with pytest.raises(FileNotFoundError, match="Tic-Tac-Toe model"):
        run_experiment.run_tictactoe(episodes=3, seed=12)


def test_tictactoe_runner_rejects_nonpositive_episodes():
    with pytest.raises(ValueError, match="greater than zero"):
        run_experiment.run_tictactoe(episodes=0, seed=12)
