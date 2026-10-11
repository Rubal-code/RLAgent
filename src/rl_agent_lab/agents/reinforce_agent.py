
import torch
from torch.distributions import Categorical

from rl_agent_lab.agents.policy_gradient import PolicyNetwork


class REINFORCEAgent:
    def __init__(
        self,
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        discount_factor=0.99,
    ):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )
        self.discount_factor = discount_factor
        self.saved_log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        )

        probabilities = self.policy(state_tensor)
        distribution = Categorical(probabilities)
        action = distribution.sample()

        self.saved_log_probs.append(
            distribution.log_prob(action)
        )

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def train_episode(self):
        returns = []
        discounted_return = 0.0

        for reward in reversed(self.rewards):
            discounted_return = (
                reward
                + self.discount_factor * discounted_return
            )
            returns.insert(0, discounted_return)

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
        )

        if len(returns) > 1:
            returns = (
                returns - returns.mean()
            ) / (returns.std(unbiased=False) + 1e-8)

        loss = torch.stack([
            -log_prob * episode_return
            for log_prob, episode_return
            in zip(self.saved_log_probs, returns)
        ]).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_log_probs.clear()
        self.rewards.clear()

        return loss.item()
