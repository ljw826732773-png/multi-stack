from __future__ import annotations

import numpy as np


class SafetyFilter:
    """Post-process a learned stack-power action with simple EMS constraints.

    The filter keeps the neural policy as the primary decision maker, but
    corrects its total fuel-cell command using SOC feedback and health-aware
    feasible limits. It is intentionally lightweight and differentiable-free so
    it can be used as an online safety layer around any policy.
    """

    def __init__(
        self,
        p_nom: float = 100.0,
        soc_ref: float = 0.60,
        soc_gain: float = 160.0,
        min_soc_guard: float = 0.55,
        max_soc_guard: float = 0.72,
        health_power: float = 1.8,
        target_alpha: float = 0.18,
    ):
        self.p_nom = p_nom
        self.soc_ref = soc_ref
        self.soc_gain = soc_gain
        self.min_soc_guard = min_soc_guard
        self.max_soc_guard = max_soc_guard
        self.health_power = health_power
        self.target_alpha = target_alpha
        self.filtered_target = 0.0

    def reset(self):
        self.filtered_target = 0.0

    def apply(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        p_dem = float(obs[0] * 300.0)
        soc = float(obs[1])
        soh = np.asarray(obs[2:6], dtype=np.float64)
        action = np.asarray(action, dtype=np.float64).reshape(4)

        limits = self.p_nom * np.clip(0.35 + 0.75 * soh, 0.20, 1.00)
        raw_power = np.clip(action, 0.0, 1.0) * self.p_nom
        raw_power = np.minimum(raw_power, limits)

        target = max(p_dem, 0.0) + self.soc_gain * (self.soc_ref - soc)
        if soc < self.min_soc_guard:
            target += self.soc_gain * (self.min_soc_guard - soc)
        if soc > self.max_soc_guard:
            target -= self.soc_gain * (soc - self.max_soc_guard)
        target = float(np.clip(target, 0.0, np.sum(limits)))
        self.filtered_target = (1.0 - self.target_alpha) * self.filtered_target + self.target_alpha * target
        if soc < self.min_soc_guard:
            self.filtered_target = max(self.filtered_target, target)
        target = float(np.clip(self.filtered_target, 0.0, np.sum(limits)))

        filtered = self._match_total(raw_power, limits, target, soh)
        return (filtered / self.p_nom).astype(np.float32)

    def _match_total(self, power: np.ndarray, limits: np.ndarray, target: float, soh: np.ndarray) -> np.ndarray:
        current = float(np.sum(power))
        if current > target and current > 1e-9:
            return power * (target / current)

        filtered = power.copy()
        for _ in range(5):
            missing = target - float(np.sum(filtered))
            if missing <= 1e-6:
                break
            room = np.maximum(limits - filtered, 0.0)
            if float(np.sum(room)) <= 1e-6:
                break
            weights = np.maximum(soh - 0.55, 0.05) ** self.health_power
            weights = weights * (room > 1e-9)
            if float(np.sum(weights)) <= 1e-9:
                weights = room
            filtered += missing * weights / float(np.sum(weights))
            filtered = np.minimum(filtered, limits)
        return np.clip(filtered, 0.0, limits)


class SafetyFilteredPolicy:
    """Policy wrapper that applies SafetyFilter to a base policy action."""

    def __init__(self, base_policy, safety_filter: SafetyFilter | None = None):
        self.base_policy = base_policy
        self.safety_filter = safety_filter or SafetyFilter()

    def reset(self):
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()
        if hasattr(self.safety_filter, "reset"):
            self.safety_filter.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.base_policy.act(obs)
        return self.safety_filter.apply(obs, action)

