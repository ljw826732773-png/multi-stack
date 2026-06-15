from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .bc import BCPolicy
from .expert import HealthAwareExpert
from .safety import SafetyFilteredPolicy


class EqualPolicy:
    def act(self, obs):
        p_dem = max(float(obs[0] * 300.0), 0.0)
        return np.full(4, np.clip(p_dem / 4.0 / 100.0, 0.0, 1.0), dtype=np.float32)


class SequentialPolicy:
    def act(self, obs):
        p_dem = max(float(obs[0] * 300.0), 0.0)
        a = np.zeros(4, dtype=np.float32)
        rem = p_dem
        for i in range(4):
            p = min(100.0, rem)
            a[i] = p / 100.0
            rem -= p
            if rem <= 0:
                break
        return a


class TorchPolicy:
    def __init__(self, path: str | Path):
        self.model = BCPolicy()
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def act(self, obs):
        return self.model.act(obs).astype(np.float32)


def default_policy_suite(
    model_path: str | Path | None = None,
    dagger_model_path: str | Path | None = None,
):
    policies = {
        "Equal": EqualPolicy(),
        "Sequential": SequentialPolicy(),
        "HC-MPC-style Expert": HealthAwareExpert(),
    }
    if model_path is not None and Path(model_path).exists():
        bc_policy = TorchPolicy(model_path)
        policies["BC Neural Policy"] = bc_policy
        policies["Safety-Filtered BC"] = SafetyFilteredPolicy(TorchPolicy(model_path))
    if dagger_model_path is not None and Path(dagger_model_path).exists():
        policies["DAgger Policy"] = TorchPolicy(dagger_model_path)
        policies["Safety-Filtered DAgger"] = SafetyFilteredPolicy(TorchPolicy(dagger_model_path))
    return policies
