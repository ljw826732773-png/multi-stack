from __future__ import annotations

import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 11, action_dim: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.forward(x).squeeze(0).cpu().numpy()

