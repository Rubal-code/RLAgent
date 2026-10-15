
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_size=8, action_size=4):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        action_probabilities = self.actor(features)
        state_value = self.critic(features)

        return action_probabilities, state_value
