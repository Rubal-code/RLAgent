import torch

from rl_agent_lab.agents.dqn import DQN
from rl_agent_lab.agents.dqn_agent import DQNAgent
from rl_agent_lab.agents.replay_buffer import ReplayBuffer


def test_dqn_output_shape():
    model = DQN(state_size=4, action_size=2)
    state = torch.zeros((1, 4))

    output = model(state)

    assert output.shape == (1, 2)


def test_replay_buffer():
    buffer = ReplayBuffer(capacity=10)

    buffer.add([0, 0, 0, 0], 0, 1, [1, 0, 0, 0], False)

    assert len(buffer) == 1
    assert len(buffer.sample(1)) == 1


def test_dqn_agent_action():
    agent = DQNAgent()
    agent.exploration_rate = 0.0

    action = agent.choose_action([0.0, 0.0, 0.0, 0.0])

    assert action in (0, 1)