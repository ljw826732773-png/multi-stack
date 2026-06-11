"""Optional SAC training entrypoint.

Install stable-baselines3 first:
    pip install stable-baselines3

This file is intentionally separate so the core behavior-cloning pipeline
remains lightweight and runnable on ordinary laptops.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from multistack_ai import MultiStackFuelCellEnv


def main():
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is not installed. Run: pip install stable-baselines3") from exc

    env = MultiStackFuelCellEnv()
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=256, gamma=0.98)
    model.learn(total_timesteps=50_000)
    model.save(ROOT / "results" / "sac_policy")


if __name__ == "__main__":
    main()

