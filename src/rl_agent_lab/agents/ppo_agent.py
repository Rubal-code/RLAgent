
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_agent_lab.agents.ppo_network import ActorCritic


class PPOAgent:
    def __init__(
        self,
        state_size=8,
        action_size=4,
        learning_rate=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    ):
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

    def choose_action(self, state):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        )

        with torch.no_grad():
            probabilities, value = self.policy(state_tensor)
            distribution = Categorical(probabilities)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        return action.item(), log_probability.item(), value.item()

    
    def compute_returns_and_advantages(
        self,
        rewards,
        values,
        dones,
        next_values,
        terminateds,
    ):
        returns = []
        advantages = []
        gae = 0.0

        for index in reversed(range(len(rewards))):
            bootstrap_mask = 1.0 - float(terminateds[index])
            continuation_mask = 1.0 - float(dones[index])

            delta = (
                rewards[index]
                + self.gamma * next_values[index] * bootstrap_mask
                - values[index]
            )

            gae = (
                delta
                + self.gamma * 0.95 * continuation_mask * gae
            )

            returns.insert(0, gae + values[index])
            advantages.insert(0, gae)

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )


    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
        epochs=10,
        batch_size=64,
    ):
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        old_log_probs = torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
        )
        returns = torch.as_tensor(returns, dtype=torch.float32)
        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
        )

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        sample_count = len(states)
        total_loss = 0.0
        update_count = 0

        for _ in range(epochs):
            indices = torch.randperm(sample_count)

            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start : start + batch_size]

                probabilities, values = self.policy(states[batch_indices])
                distribution = Categorical(probabilities)

                new_log_probs = distribution.log_prob(
                    actions[batch_indices]
                )
                entropy = distribution.entropy().mean()

                ratios = torch.exp(
                    new_log_probs - old_log_probs[batch_indices]
                )

                advantages_batch = advantages[batch_indices]

                unclipped = ratios * advantages_batch
                clipped = torch.clamp(
                    ratios,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * advantages_batch

                actor_loss = -torch.minimum(
                    unclipped,
                    clipped,
                ).mean()

                critic_loss = F.mse_loss(
                    values.squeeze(-1),
                    returns[batch_indices],
                )

                loss = (
                    actor_loss
                    + self.value_coefficient * critic_loss
                    - self.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=0.5,
                )
                self.optimizer.step()

                total_loss += loss.item()
                update_count += 1

        return total_loss / max(update_count, 1)
