from src.rl_agent_lab.environments.grid_world import GridWorld


def test_environment_reset():
    env = GridWorld(size=5)

    state = env.reset()

    assert state == (0, 0)


def test_move_right():
    env = GridWorld(size=5)

    env.reset()

    state, reward, done = env.step(3)

    assert state == (0, 1)
    assert reward == -1
    assert done is False


def test_reach_goal():
    env = GridWorld(size=2)

    env.reset()

    env.step(1)
    state, reward, done = env.step(3)

    assert state == (1, 1)
    assert reward == 10
    assert done is True