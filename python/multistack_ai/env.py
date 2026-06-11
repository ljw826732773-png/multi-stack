from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    n_stack: int = 4
    p_nom: float = 100.0
    dt: float = 1.0
    episode_len: int = 1200
    soc_ref: float = 0.60
    soc_init: float = 0.65
    soc_min: float = 0.20
    soc_max: float = 0.90
    batt_capacity_kwh: float = 52.0
    ramp_limit_kw: float = 45.0
    seed: int = 7


class MultiStackFuelCellEnv(gym.Env):
    """A lightweight Gymnasium environment for multi-stack fuel-cell EMS research.

    The action is a continuous 4-D stack-power command in normalized units.
    The environment clips physical limits, computes battery residual power,
    updates SOC and stack SOH, and returns a multi-objective reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, demand_profile: np.ndarray | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.cfg.n_stack,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(11,), dtype=np.float32)
        self.external_demand = demand_profile
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.soc = self.cfg.soc_init
        self.soh = np.array([0.95, 0.88, 0.82, 0.75], dtype=np.float64)
        self.prev_p = np.zeros(self.cfg.n_stack, dtype=np.float64)
        self.demand = self.external_demand.copy() if self.external_demand is not None else self._make_demand()
        self.history = []
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.cfg.n_stack)
        p_cmd = np.clip(action, 0.0, 1.0) * self.cfg.p_nom
        p_cmd = np.minimum(p_cmd, self.soh_safe_power_limit())
        p_cmd = np.clip(p_cmd, self.prev_p - self.cfg.ramp_limit_kw, self.prev_p + self.cfg.ramp_limit_kw)
        p_cmd = np.clip(p_cmd, 0.0, self.cfg.p_nom)

        p_dem = float(self.demand[self.t])
        p_fc = float(np.sum(p_cmd))
        p_batt = p_dem - p_fc
        self._update_soc(p_batt)
        delta_soh = self._degradation(p_cmd)
        self.soh = np.clip(self.soh - delta_soh, 0.0, 1.0)

        tracking_error = p_dem - p_fc - p_batt
        h2 = self._h2_rate(p_fc)
        start_stop = np.sum(np.abs((p_cmd > 1.0).astype(float) - (self.prev_p > 1.0).astype(float)))
        ramp = np.mean(np.abs(p_cmd - self.prev_p)) / self.cfg.p_nom
        soh_var = float(np.var(self.soh))
        soc_err = float(self.soc - self.cfg.soc_ref)

        reward = -(
            0.70 * h2
            + 2.20 * soc_err**2
            + 4.50 * soh_var
            + 0.025 * start_stop
            + 0.08 * ramp
            + 0.010 * abs(tracking_error)
        )

        self.history.append(
            {
                "p_dem": p_dem,
                "p_fc": p_fc,
                "p_batt": p_batt,
                "soc": self.soc,
                "soh": self.soh.copy(),
                "stack_power": p_cmd.copy(),
                "h2_rate": h2,
                "reward": reward,
                "start_stop": start_stop,
                "soh_var": soh_var,
            }
        )
        self.prev_p = p_cmd
        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.episode_len
        return self._obs(), float(reward), terminated, truncated, {}

    def soh_safe_power_limit(self):
        # A simple derating map: weak stacks receive lower admissible power.
        return self.cfg.p_nom * np.clip(0.35 + 0.75 * self.soh, 0.20, 1.00)

    def _obs(self):
        p_dem = float(self.demand[min(self.t, len(self.demand) - 1)])
        return np.concatenate(
            [
                np.array([p_dem / 300.0, self.soc], dtype=np.float64),
                self.soh,
                self.prev_p / self.cfg.p_nom,
                np.array([self.t / self.cfg.episode_len], dtype=np.float64),
            ]
        ).astype(np.float32)

    def _make_demand(self):
        t = np.arange(self.cfg.episode_len)
        base = 35 + 30 * np.sin(2 * np.pi * t / 420) + 15 * np.sin(2 * np.pi * t / 95)
        noise = self.rng.normal(0, 12, size=self.cfg.episode_len)
        spikes = np.zeros_like(base)
        spike_idx = self.rng.choice(self.cfg.episode_len, size=max(8, self.cfg.episode_len // 65), replace=False)
        spikes[spike_idx] = self.rng.uniform(45, 160, size=len(spike_idx))
        regen_idx = self.rng.choice(self.cfg.episode_len, size=max(5, self.cfg.episode_len // 90), replace=False)
        spikes[regen_idx] -= self.rng.uniform(35, 110, size=len(regen_idx))
        demand = base + noise + spikes
        return np.clip(demand, -120, 280).astype(np.float64)

    def _update_soc(self, p_batt_kw: float):
        # Positive battery power means discharge.
        delta = -(p_batt_kw * self.cfg.dt / 3600.0) / self.cfg.batt_capacity_kwh
        self.soc = float(np.clip(self.soc + delta, self.cfg.soc_min, self.cfg.soc_max))

    def _degradation(self, p_cmd: np.ndarray):
        k_trans = 5.93e-7
        k_start = 1.96e-7
        k_load = 1.20e-8
        trans = k_trans * np.abs(p_cmd - self.prev_p) / self.cfg.p_nom
        start = k_start * np.abs((p_cmd > 1.0).astype(float) - (self.prev_p > 1.0).astype(float))
        load = k_load * np.maximum(p_cmd / self.cfg.p_nom, 0.10)
        return trans + start + load

    def _h2_rate(self, p_fc_kw: float):
        if p_fc_kw <= 1e-6:
            return 0.0
        # U-shaped specific consumption approximation.
        specific = 0.055 + 0.10 / (p_fc_kw + 8.0) + 0.000035 * max(p_fc_kw - 190.0, 0.0) ** 2 / 100.0
        return specific * p_fc_kw / 3600.0

