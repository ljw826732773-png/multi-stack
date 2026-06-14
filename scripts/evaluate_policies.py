import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import HealthAwareExpert
from multistack_ai.evaluate import rollout
from multistack_ai.policies import EqualPolicy, SequentialPolicy, TorchPolicy


class _PolicySuite:
    @staticmethod
    def build(model_path):
        policies = {
            "Equal": EqualPolicy(),
            "Sequential": SequentialPolicy(),
            "HC-MPC-style Expert": HealthAwareExpert(),
        }
        if model_path.exists():
            policies["BC Neural Policy"] = TorchPolicy(model_path)
        return policies


def aggregate(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    policies = _PolicySuite.build(ROOT / "results" / "bc_policy.pt")

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