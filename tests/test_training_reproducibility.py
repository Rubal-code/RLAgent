
from rl_agent_lab.training.train_dqn import train as train_dqn
from rl_agent_lab.training.train_reinforce import train as train_reinforce


def test_dqn_training_reproduces_rewards_with_same_seed():
    _, first_rewards = train_dqn(episodes=3, seed=123)
    _, second_rewards = train_dqn(episodes=3, seed=123)

    assert first_rewards == second_rewards


def test_reinforce_training_reproduces_rewards_with_same_seed():
    _, first_rewards = train_reinforce(episodes=3, seed=123)
    _, second_rewards = train_reinforce(episodes=3, seed=123)

    assert first_rewards == second_rewards
