
import gymnasium as gym
import torch

from rl_agent_lab.agents.reinforce_agent import REINFORCEAgent
from rl_agent_lab.agents.reinforce_model_io import load_model


def evaluate(episodes=10):
    env = gym.make("CartPole-v1")
    agent = REINFORCEAgent()

    load_model(agent, "models/cartpole_reinforce.pth")
    agent.policy.eval()

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
            )

            with torch.no_grad():
                probabilities = agent.policy(state_tensor)
                action = torch.argmax(probabilities).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.0f}")

    average_reward = sum(rewards) / len(rewards)
    print(f"\nAverage reward over {episodes} episodes: {average_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
