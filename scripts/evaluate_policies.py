import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import HealthAwareExpert
from multistack_ai.bc import BCPolicy
from multistack_ai.evaluate import rollout


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
    def __init__(self, path):
        self.model = BCPolicy()
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def act(self, obs):
        return self.model.act(obs).astype(np.float32)


def aggregate(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    policies = {
        "Equal": EqualPolicy(),
        "Sequential": SequentialPolicy(),
        "HC-MPC-style Expert": HealthAwareExpert(),
    }
    model_path = ROOT / "results" / "bc_policy.pt"
    if model_path.exists():
        policies["BC Neural Policy"] = TorchPolicy(model_path)

    out = ROOT / "results" / "policy_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for name, policy in policies.items():
        summary = aggregate(rollout(policy, episodes=8, seed=2026))
        summary["policy"] = name
        records.append(summary)
        print(name, summary)

    keys = ["policy"] + [k for k in records[0].keys() if k != "policy"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"saved comparison to {out}")


if __name__ == "__main__":
    main()

