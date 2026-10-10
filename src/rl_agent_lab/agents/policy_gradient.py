import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)