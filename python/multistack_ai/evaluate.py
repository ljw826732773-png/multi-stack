from __future__ import annotations

import numpy as np

from .env import MultiStackFuelCellEnv


def rollout(policy, episodes: int = 5, seed: int = 0):
    rows = []
    for ep in range(episodes):
        env = MultiStackFuelCellEnv()
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        rows.append(summarize(env))
    return rows


def summarize(env: MultiStackFuelCellEnv):
    hist = env.history
    h2 = float(np.sum([h["h2_rate"] for h in hist]))
    soh = hist[-1]["soh"]
    soc = np.array([h["soc"] for h in hist])
    p_dem = np.array([h["p_dem"] for h in hist])
    p_fc = np.array([h["p_fc"] for h in hist])
    starts = float(np.sum([h["start_stop"] for h in hist]))
    return {
        "h2_proxy_kg": h2,
        "final_soh_range": float(np.max(soh) - np.min(soh)),
        "final_soh_var": float(np.var(soh)),
        "final_mean_soh": float(np.mean(soh)),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "power_mae_kw": float(np.mean(np.abs(p_dem - p_fc))),
        "start_stop_count": starts,
    }

