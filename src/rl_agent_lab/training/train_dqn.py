
import random

import gymnasium as gym
import numpy as np
import torch

from rl_agent_lab.agents.dqn_agent import DQNAgent
from rl_agent_lab.agents.dqn_model_io import save_model
from rl_agent_lab.visualization.plot_rewards import plot_rewards


def train(episodes=500, seed=None):
    """Train a DQN agent with optional reproducible random seeding."""
    if episodes <= 0:
        raise ValueError("Episodes must be greater than zero.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = gym.make("CartPole-v1")

    if seed is not None:
        env.action_space.seed(seed)

    agent = DQNAgent()
    rewards = []

    try:
        for episode in range(episodes):
            if episode == 0 and seed is not None:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            total_reward = 0.0

            for _ in range(500):
                action = agent.choose_action(state)

                next_state, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                done = terminated or truncated

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                agent.train_step()

                state = next_state
                total_reward += reward

                if done:
                    break

            agent.update_target_network()
            agent.decay_exploration()
            rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                print(
                    f"Episode: {episode + 1}, "
                    f"Reward: {total_reward:.0f}, "
                    f"Epsilon: {agent.exploration_rate:.3f}"
                )
    finally:
        env.close()

    return agent, rewards


if __name__ == "__main__":
    agent, rewards = train(episodes=500, seed=42)

    save_model(
        agent,
        "models/cartpole_dqn.pth",
    )

    plot_rewards(rewards)
