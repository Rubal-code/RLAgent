import gymnasium as gym

from rl_agent_lab.agents.q_learning import QLearningAgent


def train(episodes=1000):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
    )

    agent = QLearningAgent()

    successes = 0

    for episode in range(episodes):
        state, _ = env.reset()

        for _ in range(100):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            if done:
                if reward == 1:
                    successes += 1
                break

        agent.decay_exploration()

        if (episode + 1) % 100 == 0:
            success_rate = successes / (episode + 1) * 100

            print(
                f"Episode: {episode + 1}, "
                f"Success Rate: {success_rate:.1f}%, "
                f"Epsilon: {agent.exploration_rate:.3f}"
            )

    env.close()

    return agent


if __name__ == "__main__":
    train(episodes=1000)