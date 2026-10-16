
import pytest

from rl_agent_lab.environments.tictactoe import TicTacToe


def test_reset_and_legal_actions():
    env = TicTacToe()

    state = env.reset()

    assert state == (0,) * 9
    assert env.legal_actions() == list(range(9))


def test_agent_win():
    env = TicTacToe()

    env.step(0, player=TicTacToe.AGENT)
    env.step(3, player=TicTacToe.HUMAN)
    env.step(1, player=TicTacToe.AGENT)
    env.step(4, player=TicTacToe.HUMAN)

    _, reward, terminated, info = env.step(
        2, player=TicTacToe.AGENT
    )

    assert reward == 1.0
    assert terminated is True
    assert info["winner"] == TicTacToe.AGENT


def test_draw():
    env = TicTacToe()

    # X O X
    # X O O
    # O X X
    moves = [
        (0, TicTacToe.HUMAN),
        (1, TicTacToe.AGENT),
        (2, TicTacToe.HUMAN),
        (4, TicTacToe.AGENT),
        (3, TicTacToe.HUMAN),
        (5, TicTacToe.AGENT),
        (7, TicTacToe.HUMAN),
        (6, TicTacToe.AGENT),
        (8, TicTacToe.HUMAN),
    ]

    for action, player in moves:
        _, _, terminated, _ = env.step(action, player=player)

    assert terminated is True
    assert env.is_draw()


def test_illegal_move_raises_error():
    env = TicTacToe()
    env.step(0, player=TicTacToe.AGENT)

    with pytest.raises(ValueError):
        env.step(0, player=TicTacToe.HUMAN)
