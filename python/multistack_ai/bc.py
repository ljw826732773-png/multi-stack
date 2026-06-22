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


class SequenceBCPolicy(nn.Module):
    """GRU behavior-cloning policy for history-dependent EMS control."""

    def __init__(self, obs_dim: int = 11, action_dim: int = 4, hidden: int = 96, layers: int = 1):
        super().__init__()
        self.hidden_size = hidden
        self.layers = layers
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),
        )
        self._hidden = None

    def forward(self, x, hidden=None):
        z = self.encoder(x)
        out, hidden = self.gru(z, hidden)
        return self.head(out), hidden

    def reset(self):
        self._hidden = None

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        action, self._hidden = self.forward(x, self._hidden)
        self._hidden = self._hidden.detach()
        return action.squeeze(0).squeeze(0).cpu().numpy()
