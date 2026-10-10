import gymnasium as gym

from rl_agent_lab.agents.dqn_agent import DQNAgent
from rl_agent_lab.agents.dqn_model_io import load_model


def evaluate(episodes=10):
    env = gym.make("CartPole-v1")
    agent = DQNAgent()

    load_model(agent, "models/cartpole_dqn.pth")

    agent.exploration_rate = 0.0
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action = agent.choose_action(state)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        scores.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.0f}")

    env.close()

    average_reward = sum(scores) / len(scores)
    print(f"\nAverage reward: {average_reward:.2f}")

    return average_reward


if __name__ == "__main__":
    evaluate()