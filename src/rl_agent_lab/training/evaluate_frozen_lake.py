import gymnasium as gym

from rl_agent_lab.training.train_frozen_lake import train

ACTION_ARROWS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}


def print_policy(agent):
    print("\n--- Learned FrozenLake Policy ---")

    for row in range(4):
        line = ""

        for col in range(4):
            state = row * 4 + col

            if state == 0:
                line += "S "
            elif state == 15:
                line += "G "
            elif state in {5, 7, 11, 12}:
                line += "H "
            else:
                q_values = agent.get_q_values(state)
                best_action = q_values.index(max(q_values))
                line += f"{ACTION_ARROWS[best_action]} "

        print(line)


def evaluate(agent, episodes=100):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
    )

    successes = 0

    for _ in range(episodes):
        state, _ = env.reset()

        for _ in range(100):
            q_values = agent.get_q_values(state)
            action = q_values.index(max(q_values))

            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state

            if terminated or truncated:
                if reward == 1:
                    successes += 1
                break

    env.close()

    success_rate = successes / episodes * 100

    print(f"\nEvaluation Success Rate: {success_rate:.1f}%")

    return success_rate


if __name__ == "__main__":
    agent = train(episodes=5000)

    print_policy(agent)

    evaluate(
        agent,
        episodes=100,
    )