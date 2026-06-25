import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import EnvConfig, HealthAwareExpert, MultiStackFuelCellEnv, available_cycles, make_cycle_demand


def collect_cycle(cycle: str, seed: int):
    demand = make_cycle_demand(cycle)
    env = MultiStackFuelCellEnv(EnvConfig(episode_len=len(demand), seed=seed), demand_profile=demand)
    expert = HealthAwareExpert()
    obs, _ = env.reset(seed=seed)
    expert.reset()
    xs, ys = [], []
    done = False
    while not done:
        action = expert.act(obs)
        xs.append(obs.copy())
        ys.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="*", default=available_cycles())
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "epa_expert_dataset.npz")
    args = parser.parse_args()

    all_x, all_y, lengths, names = [], [], [], []
    for rep in range(args.repeat):
        for cycle in args.cycles:
            x, y = collect_cycle(cycle, seed=5000 + rep * 100 + len(names))
            all_x.append(x)
            all_y.append(y)
            lengths.append(len(x))
            names.append(cycle)
            print(f"collected cycle={cycle} repeat={rep + 1} samples={len(x)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=np.concatenate(all_x).astype(np.float32),
        Y=np.concatenate(all_y).astype(np.float32),
        episode_lengths=np.asarray(lengths, dtype=np.int32),
        cycle_names=np.asarray(names),
    )
    print(f"saved {sum(lengths)} EPA expert samples to {args.out}")


if __name__ == "__main__":
    main()
