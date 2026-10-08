import gymnasium as gym

from rl_agent_lab.agents.dqn_agent import DQNAgent
from rl_agent_lab.visualization.plot_rewards import plot_rewards


def train(episodes=500):
    env = gym.make("CartPole-v1")

    agent = DQNAgent()

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

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

    env.close()

    return agent, rewards


if __name__ == "__main__":
    agent, rewards = train(episodes=500)
    plot_rewards(rewards)