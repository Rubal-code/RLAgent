from rl_agent_lab.agents.bellman import bellman_update


def test_bellman_update():
    new_q = bellman_update(
        current_q=2.0,
        reward=1.0,
        next_max_q=4.0,
        learning_rate=0.1,
        discount_factor=0.9,
        done=False,
    )

    expected = 2.26

    assert abs(new_q - expected) < 1e-6


def test_bellman_update_terminal_state():
    new_q = bellman_update(
        current_q=2.0,
        reward=10.0,
        next_max_q=100.0,
        learning_rate=0.1,
        discount_factor=0.9,
        done=True,
    )

    expected = 2.8

    assert abs(new_q - expected) < 1e-6
