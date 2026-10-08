import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_agent_lab.agents.dqn import DQN
from rl_agent_lab.agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        self.device = torch.device("cpu")

        self.policy_network = DQN(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network = DQN(
            state_size,
            action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.memory = ReplayBuffer(capacity=10000)

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        current_q_values = self.policy_network(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(
                dim=1
            ).values

            target_q_values = rewards + (
                1 - dones
            ) * self.discount_factor * next_q_values

        loss = F.mse_loss(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate,
        )