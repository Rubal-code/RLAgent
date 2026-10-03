from src.rl_agent_lab.agents.q_learning import QLearningAgent
from src.rl_agent_lab.environments.grid_world import GridWorld
from src.rl_agent_lab.training.train_q_learning import train


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


def evaluate(agent: QLearningAgent, env: GridWorld):
    """
    Run the trained agent without exploration.
    """

    state = env.reset()

    path = [state]

    while True:
        q_values = agent.get_q_values(state)

        # Always choose the best learned action.
        action = q_values.index(max(q_values))

        next_state, reward, done = env.step(action)

        path.append(next_state)

        print(
            f"State: {state} | "
            f"Action: {ACTION_NAMES[action]} | "
            f"Reward: {reward} | "
            f"Next: {next_state}"
        )

        state = next_state

        if done:
            break

    return path

def print_q_table(agent):
    print("\n--- Q-Table ---")

    for state, q_values in sorted(agent.q_table.items()):
        print(
            f"{state}: "
            f"UP={q_values[0]:.2f}, "
            f"DOWN={q_values[1]:.2f}, "
            f"LEFT={q_values[2]:.2f}, "
            f"RIGHT={q_values[3]:.2f}"
        )


if __name__ == "__main__":
    agent, _ = train(episodes=1000)

    env = GridWorld(size=5)

    print_q_table(agent)

    print("\n--- Evaluation ---")

    path = evaluate(agent, env)

    print("\nLearned path:")
    print(" -> ".join(map(str, path)))