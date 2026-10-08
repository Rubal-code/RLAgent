from pathlib import Path

import torch

from rl_agent_lab.agents.dqn_agent import DQNAgent


def save_model(agent: DQNAgent, path: str):
    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        agent.policy_network.state_dict(),
        file_path,
    )


def load_model(agent: DQNAgent, path: str):
    file_path = Path(path)

    state_dict = torch.load(
        file_path,
        map_location=agent.device,
    )

    agent.policy_network.load_state_dict(state_dict)

    agent.update_target_network()

    return agent
