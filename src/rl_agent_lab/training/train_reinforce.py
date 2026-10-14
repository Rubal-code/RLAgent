
import gymnasium as gym
import matplotlib.pyplot as plt

from rl_agent_lab.agents.reinforce_agent import REINFORCEAgent
from rl_agent_lab.agents.reinforce_model_io import save_model


def train(episodes=1000):
    env = gym.make("CartPole-v1")
    agent = REINFORCEAgent()

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            agent.store_reward(reward)
            total_reward += reward
            state = next_state

        loss = agent.train_episode()
        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            recent_average = sum(episode_rewards[-50:]) / 50

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Average Reward: {recent_average:.2f} | "
                f"Loss: {loss:.4f}"
            )

    env.close()
    return agent, episode_rewards


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode reward")

    if len(rewards) >= 50:
        moving_averages = [
            sum(rewards[i - 49 : i + 1]) / 50
            for i in range(49, len(rewards))
        ]
        plt.plot(
            range(49, len(rewards)),
            moving_averages,
            label="50-episode average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Training on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trained_agent, rewards = train(episodes=1000)
    save_model(trained_agent, "models/cartpole_reinforce.pth")
    plot_rewards(rewards)