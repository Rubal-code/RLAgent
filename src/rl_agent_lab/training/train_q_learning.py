from src.rl_agent_lab.agents.q_learning import QLearningAgent
from src.rl_agent_lab.environments.grid_world import GridWorld


def train(episodes=1000):
    env = GridWorld(size=5)

    agent = QLearningAgent()

    rewards = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        while True:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            agent.update(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            total_reward += reward

            if done:
                break

        agent.decay_exploration()

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode: {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Epsilon: {agent.exploration_rate:.3f}"
            )

    return agent, rewards


if __name__ == "__main__":
    train()