
from pathlib import Path

import torch

from rl_agent_lab.agents.reinforce_agent import REINFORCEAgent


def save_model(agent: REINFORCEAgent, path: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(agent.policy.state_dict(), file_path)


def load_model(agent: REINFORCEAgent, path: str) -> REINFORCEAgent:
    file_path = Path(path)

    state_dict = torch.load(
        file_path,
        map_location="cpu",
        weights_only=True,
    )

    agent.policy.load_state_dict(state_dict)
    return agent
