
from rl_agent_lab.agents.reinforce_agent import REINFORCEAgent


def test_reinforce_agent_choose_action():
    agent = REINFORCEAgent()
    action = agent.choose_action([0.0, 0.0, 0.0, 0.0])

    assert action in (0, 1)
    assert len(agent.saved_log_probs) == 1


def test_reinforce_agent_train_episode():
    agent = REINFORCEAgent()

    agent.choose_action([0.0, 0.0, 0.0, 0.0])
    agent.store_reward(1.0)

    loss = agent.train_episode()

    assert isinstance(loss, float)
    assert len(agent.saved_log_probs) == 0
    assert len(agent.rewards) == 0
