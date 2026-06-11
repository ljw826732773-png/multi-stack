from __future__ import annotations

import numpy as np


class HealthAwareExpert:
    """Heuristic HC-MPC-style expert policy.

    It smooths the fuel-cell command, corrects SOC deviation, and allocates
    stack power with health-aware asymmetric weights. It is intentionally
    lightweight so it can generate expert data without MATLAB or quadprog.
    """

    def __init__(self, p_nom: float = 100.0, alpha: float = 0.06, soc_ref: float = 0.60):
        self.p_nom = p_nom
        self.alpha = alpha
        self.soc_ref = soc_ref
        self.filtered = 0.0

    def reset(self):
        self.filtered = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        p_dem = float(obs[0] * 300.0)
        soc = float(obs[1])
        soh = np.asarray(obs[2:6], dtype=np.float64)
        prev_p = np.asarray(obs[6:10], dtype=np.float64) * self.p_nom

        self.filtered = (1.0 - self.alpha) * self.filtered + self.alpha * max(p_dem, 0.0)
        soc_comp = -180.0 * (soc - self.soc_ref)
        p_fc = float(np.clip(self.filtered + soc_comp, 0.0, 4 * self.p_nom))

        # Allocate more load to healthier stacks while derating weak stacks.
        weights = np.maximum(soh - 0.55, 0.05) ** 2.2
        limits = self.p_nom * np.clip(0.35 + 0.75 * soh, 0.20, 1.00)
        raw = p_fc * weights / max(np.sum(weights), 1e-9)
        p = np.minimum(raw, limits)

        # Redistribute remaining feasible power.
        for _ in range(4):
            rem = p_fc - np.sum(p)
            if rem <= 1e-6:
                break
            room = np.maximum(limits - p, 0.0)
            if np.sum(room) <= 1e-6:
                break
            p += rem * room / np.sum(room)
            p = np.minimum(p, limits)

        p = np.clip(p, prev_p - 45.0, prev_p + 45.0)
        p = np.clip(p, 0.0, self.p_nom)
        return (p / self.p_nom).astype(np.float32)

