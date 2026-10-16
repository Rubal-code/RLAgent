
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_agent_lab.agents.ppo_agent import PPOAgent


def train(episodes=1000, update_interval=2048):
    env = gym.make("LunarLander-v3")
    agent = PPOAgent()

    checkpoint_path = Path("models/lunarlander_ppo.pth")

    if checkpoint_path.exists():
        agent.policy.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        agent.policy.train()
        print(f"Resuming training from {checkpoint_path}")

    states = []
    actions = []
    rewards = []
    dones = []
    terminateds = []
    log_probs = []
    values = []
    next_values = []

    episode_rewards = []
    episode_reward = 0.0
    completed_episodes = 0

    state, _ = env.reset()

    while completed_episodes < episodes:
        action, log_prob, value = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        with torch.no_grad():
            _, next_value_tensor = agent.policy(
                torch.as_tensor(next_state, dtype=torch.float32)
            )
            next_value = next_value_tensor.item()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        terminateds.append(terminated)
        log_probs.append(log_prob)
        values.append(value)
        next_values.append(next_value)

        episode_reward += reward
        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            completed_episodes += 1

            if completed_episodes % 10 == 0:
                average_reward = float(np.mean(episode_rewards[-10:]))
                print(
                    f"Episode {completed_episodes}/{episodes} | "
                    f"Average reward (last 10): {average_reward:.2f}"
                )

            episode_reward = 0.0
            state, _ = env.reset()

        if len(states) >= update_interval:
            returns, advantages = agent.compute_returns_and_advantages(
                rewards=rewards,
                values=values,
                dones=dones,
                next_values=next_values,
                terminateds=terminateds,
            )

            agent.update(
                states=np.asarray(states, dtype=np.float32),
                actions=actions,
                old_log_probs=log_probs,
                returns=returns,
                advantages=advantages,
            )

            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()
            terminateds.clear()
            log_probs.clear()
            values.clear()
            next_values.clear()

    if states:
        returns, advantages = agent.compute_returns_and_advantages(
            rewards=rewards,
            values=values,
            dones=dones,
            next_values=next_values,
            terminateds=terminateds,
        )

        agent.update(
            states=np.asarray(states, dtype=np.float32),
            actions=actions,
            old_log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        )

    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    trained_agent, rewards = train(episodes=1000)

    model_path = Path("models/lunarlander_ppo.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_agent.policy.state_dict(), model_path)

    print(f"Model saved to {model_path}")
    print(f"Completed episodes: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
