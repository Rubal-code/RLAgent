
import gymnasium as gym
import torch

from rl_agent_lab.agents.ppo_network import ActorCritic


def evaluate(episodes=10, model_path="models/lunarlander_ppo.pth"):
    env = gym.make("LunarLander-v3")
    agent = ActorCritic()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    agent.load_state_dict(state_dict)
    agent.eval()

    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
            )

            with torch.no_grad():
                probabilities, _ = agent(state_tensor)
                action = torch.argmax(probabilities).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    print(f"\nAverage reward: {sum(rewards) / len(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    evaluate()
