
import pytest

from rl_agent_lab.agents.tictactoe_agent import TicTacToeAgent


def test_agent_selects_only_legal_actions():
    agent = TicTacToeAgent(epsilon=0.0)
    state = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    legal_actions = [0, 2, 4]

    for _ in range(20):
        assert agent.select_action(
            state, legal_actions, training=False
        ) in legal_actions


def test_q_learning_update_for_terminal_reward():
    agent = TicTacToeAgent(learning_rate=0.5, epsilon=0.0)
    state = (0,) * 9

    agent.update(
        state=state,
        action=4,
        reward=1.0,
        next_state=state,
        next_legal_actions=[],
        terminated=True,
    )

    assert agent.q_table[state][4] == pytest.approx(0.5)


def test_q_learning_update_uses_future_value():
    agent = TicTacToeAgent(
        learning_rate=1.0,
        discount_factor=0.9,
        epsilon=0.0,
    )

    state = (0,) * 9
    next_state = (1, 0, 0, 0, -1, 0, 0, 0, 0)

    agent.q_table[next_state] = {1: 0.8, 2: 0.4}

    agent.update(
        state=state,
        action=0,
        reward=0.1,
        next_state=next_state,
        next_legal_actions=[1, 2],
        terminated=False,
    )

    assert agent.q_table[state][0] == pytest.approx(0.82)


def test_epsilon_decay_respects_minimum():
    agent = TicTacToeAgent(
        epsilon=0.1,
        epsilon_decay=0.5,
        min_epsilon=0.05,
    )

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)
