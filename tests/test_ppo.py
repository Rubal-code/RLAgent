
import numpy as np
import torch

from rl_agent_lab.agents.ppo_agent import PPOAgent


def test_ppo_choose_action():
    agent = PPOAgent()
    action, log_probability, value = agent.choose_action(
        np.zeros(8, dtype=np.float32)
    )

    assert action in (0, 1, 2, 3)
    assert isinstance(log_probability, float)
    assert isinstance(value, float)


def test_ppo_returns_and_advantages():
    agent = PPOAgent()

    returns, advantages = agent.compute_returns_and_advantages(
        rewards=[1.0, 1.0],
        values=[0.5, 0.6],
        dones=[False, True],
    )

    assert returns.shape == (2,)
    assert advantages.shape == (2,)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()
    assert returns[1].item() == 1.0


def test_ppo_update():
    agent = PPOAgent()

    states = np.zeros((4, 8), dtype=np.float32)
    actions = [0, 1, 2, 3]
    old_log_probs = [
        agent.choose_action(state)[1] for state in states
    ]

    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    advantages = torch.tensor([0.1, 0.2, 0.3, 0.4])

    loss = agent.update(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
        epochs=2,
        batch_size=2,
    )

    assert isinstance(loss, float)
    assert np.isfinite(loss)
