import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNPolicy(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy_input = torch.zeros(1, 4, 84, 84)
        conv_out = self.conv(dummy_input)
        conv_out_size = conv_out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, action_dim)

        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value


# testing code

# model = CNNPolicy()

# state = torch.randn(1, 4, 84, 84)
# logits, value = model(state)

# print("Action logits:", logits)
# print("State value:", value)
