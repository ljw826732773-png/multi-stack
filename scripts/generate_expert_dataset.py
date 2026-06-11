import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import HealthAwareExpert, MultiStackFuelCellEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "expert_dataset.npz")
    args = parser.parse_args()

    xs, ys = [], []
    for ep in range(args.episodes):
        env = MultiStackFuelCellEnv()
        expert = HealthAwareExpert()
        obs, _ = env.reset(seed=1000 + ep)
        expert.reset()
        done = False
        while not done:
            action = expert.act(obs)
            xs.append(obs.copy())
            ys.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=np.asarray(xs, dtype=np.float32), Y=np.asarray(ys, dtype=np.float32))
    print(f"saved {len(xs)} expert samples to {args.out}")


if __name__ == "__main__":
    main()

