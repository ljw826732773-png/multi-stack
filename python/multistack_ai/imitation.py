from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .bc import BCPolicy
from .env import EnvConfig, MultiStackFuelCellEnv
from .evaluate import summarize
from .expert import HealthAwareExpert


@dataclass
class TrainResult:
    model: BCPolicy
    train_mse: float
    val_mse: float


def collect_expert_data(episodes: int, seed: int = 1000, episode_len: int = 1200):
    xs, ys = [], []
    for ep in range(episodes):
        env = MultiStackFuelCellEnv(EnvConfig(episode_len=episode_len))
        expert = HealthAwareExpert()
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        while not done:
            action = expert.act(obs)
            xs.append(obs.copy())
            ys.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect_dagger_queries(policy, episodes: int, seed: int = 2000, episode_len: int = 1200):
    """Roll out the current learner, but label visited states with the expert."""

    xs, ys = [], []
    summaries = []
    for ep in range(episodes):
        env = MultiStackFuelCellEnv(EnvConfig(episode_len=episode_len))
        expert = HealthAwareExpert()
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            expert_action = expert.act(obs)
            learner_action = policy.act(obs)
            xs.append(obs.copy())
            ys.append(expert_action.copy())
            obs, _, terminated, truncated, _ = env.step(learner_action)
            done = terminated or truncated
        summaries.append(summarize(env))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), summaries


def split_dataset(x, y, val_fraction: float = 0.15, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


def make_sequence_dataset(
    x,
    y,
    episode_len: int | None = 1200,
    seq_len: int = 32,
    stride: int = 8,
    episode_lengths: list[int] | np.ndarray | None = None,
):
    """Build rolling windows from contiguous expert trajectories."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if episode_lengths is None:
        if episode_len is None:
            raise ValueError("episode_len is required when episode_lengths is not provided")
        n_episodes = len(x) // episode_len
        if n_episodes < 1:
            raise ValueError("dataset is shorter than one episode")
        episode_lengths = np.full(n_episodes, episode_len, dtype=int)
    else:
        episode_lengths = np.asarray(episode_lengths, dtype=int)
        if int(np.sum(episode_lengths)) > len(x):
            raise ValueError("episode_lengths exceed dataset length")

    xs, ys = [], []
    start = 0
    for length in episode_lengths:
        end = start + int(length)
        ep_x = x[start:end]
        ep_y = y[start:end]
        if len(ep_x) < seq_len:
            start = end
            continue
        for offset in range(0, len(ep_x) - seq_len + 1, stride):
            xs.append(ep_x[offset : offset + seq_len])
            ys.append(ep_y[offset : offset + seq_len])
        start = end
    if not xs:
        raise ValueError("no sequence windows could be built")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_bc_model(
    x,
    y,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 7,
    model: BCPolicy | None = None,
):
    torch.manual_seed(seed)
    x_train, x_val, y_train, y_val = split_dataset(x, y, seed=seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model or BCPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    train_mse = 0.0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        train_mse = total / len(x_train)
    model.eval()
    with torch.no_grad():
        val_mse = float(loss_fn(model(torch.tensor(x_val)), torch.tensor(y_val)).item())
    return TrainResult(model=model, train_mse=train_mse, val_mse=val_mse)


def mean_summary(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
