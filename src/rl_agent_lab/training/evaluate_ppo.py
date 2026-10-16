
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from rl_agent_lab.agents.ppo_network import ActorCritic


def evaluate(episodes=30, stochastic=False):
    env = gym.make("LunarLander-v3")
    agent = ActorCritic()

    state_dict = torch.load(
        "models/lunarlander_ppo.pth",
        map_location="cpu",
        weights_only=True,
    )
    agent.load_state_dict(state_dict)
    agent.eval()

    rewards = []

    for episode in range(episodes):
        seed = 1000 + episode
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        terminated = False
        truncated = False

        torch.manual_seed(seed)

        while not (terminated or truncated):
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
            )

            with torch.no_grad():
                probabilities, _ = agent(state_tensor)

                if stochastic:
                    action = Categorical(probabilities).sample().item()
                else:
                    action = torch.argmax(probabilities).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

        print(
            f"Episode {episode + 1}/{episodes}: "
            f"Reward = {total_reward:.2f}"
        )

    average_reward = float(np.mean(rewards))
    print(f"\nPolicy: {'Stochastic' if stochastic else 'Greedy'}")
    print(f"Average reward over {episodes} episodes: {average_reward:.2f}")

    env.close()
    return average_reward


if __name__ == "__main__":
    greedy_average = evaluate(episodes=30, stochastic=False)
    stochastic_average = evaluate(episodes=30, stochastic=True)

    print("\n--- Policy Comparison ---")
    print(f"Greedy average:     {greedy_average:.2f}")
    print(f"Stochastic average: {stochastic_average:.2f}")
