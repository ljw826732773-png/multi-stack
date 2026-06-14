import numpy as np

from multistack_ai.safety import SafetyFilter, SafetyFilteredPolicy


class ConstantPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def act(self, obs):
        return self.action


def make_obs(p_dem=120.0, soc=0.60):
    soh = np.array([0.95, 0.88, 0.82, 0.75], dtype=np.float32)
    prev = np.zeros(4, dtype=np.float32)
    return np.concatenate([[p_dem / 300.0, soc], soh, prev, [0.0]]).astype(np.float32)


def test_safety_filter_returns_normalized_action():
    filt = SafetyFilter()
    obs = make_obs()
    action = filt.apply(obs, np.ones(4, dtype=np.float32) * 2.0)

    assert action.shape == (4,)
    assert np.all(action >= 0.0)
    assert np.all(action <= 1.0)


def test_safety_filter_increases_power_when_soc_is_low():
    filt = SafetyFilter(target_alpha=1.0)
    low_soc = make_obs(p_dem=60.0, soc=0.52)
    ref_soc = make_obs(p_dem=60.0, soc=0.60)
    base_action = np.ones(4, dtype=np.float32) * 0.05

    low_power = np.sum(filt.apply(low_soc, base_action))
    filt.reset()
    ref_power = np.sum(filt.apply(ref_soc, base_action))

    assert low_power > ref_power


def test_safety_filtered_policy_resets_base_and_filter():
    base = ConstantPolicy([0.1, 0.1, 0.1, 0.1])
    policy = SafetyFilteredPolicy(base)

    policy.safety_filter.filtered_target = 10.0
    policy.reset()

    assert base.reset_count == 1
    assert policy.safety_filter.filtered_target == 0.0
