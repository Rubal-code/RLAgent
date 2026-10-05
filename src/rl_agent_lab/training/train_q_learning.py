from rl_agent_lab.agents.model_io import save_q_table
from rl_agent_lab.agents.q_learning import QLearningAgent
from rl_agent_lab.environments.grid_world import GridWorld
from rl_agent_lab.visualization.plot_rewards import plot_rewards


def train(episodes=1000):
    env = GridWorld(size=5)
    agent = QLearningAgent()

    rewards = []
    successes = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        success = False

        for _ in range(100):
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
                success = True
                break

        agent.decay_exploration()

        rewards.append(total_reward)
        successes.append(success)

        if (episode + 1) % 100 == 0:
            success_rate = sum(successes[-100:]) / 100 * 100

            print(
                f"Episode: {episode + 1}, "
                f"Reward: {total_reward}, "
                f"Success Rate: {success_rate:.1f}%, "
                f"Epsilon: {agent.exploration_rate:.3f}"
            )

    return agent, rewards, successes


if __name__ == "__main__":
    agent, rewards, successes = train(episodes=1000)

    save_q_table(
        agent,
        "models/q_table.json",
    )

    plot_rewards(rewards)