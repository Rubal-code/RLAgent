import json
from pathlib import Path

from rl_agent_lab.agents.q_learning import QLearningAgent


def save_q_table(agent: QLearningAgent, path: str):
    """Save Q-table to a JSON file."""

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        f"{state[0]},{state[1]}": q_values
        for state, q_values in agent.q_table.items()
    }

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_q_table(agent: QLearningAgent, path: str):
    """Load Q-table from a JSON file."""

    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    agent.q_table = {
        tuple(map(int, state.split(","))): q_values
        for state, q_values in data.items()
    }

    return agent