import numpy as np

from multistack_ai import EnvConfig, MultiStackFuelCellEnv


def test_env_reset_and_step_shapes():
    env = MultiStackFuelCellEnv(EnvConfig(episode_len=5))
    obs, info = env.reset(seed=123)

    assert info == {}
    assert obs.shape == (11,)
    assert env.action_space.shape == (4,)

    next_obs, reward, terminated, truncated, info = env.step(np.ones(4, dtype=np.float32) * 0.2)

    assert next_obs.shape == (11,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert len(env.history) == 1


def test_env_respects_episode_length():
    env = MultiStackFuelCellEnv(EnvConfig(episode_len=3))
    obs, _ = env.reset(seed=1)

    truncated = False
    for _ in range(3):
        obs, _, _, truncated, _ = env.step(np.zeros(4, dtype=np.float32))

    assert truncated is True
    assert len(env.history) == 3


def test_env_clips_to_physical_limits():
    env = MultiStackFuelCellEnv(EnvConfig(episode_len=2, ramp_limit_kw=20.0))
    env.reset(seed=1)

    env.step(np.ones(4, dtype=np.float32) * 10.0)
    stack_power = env.history[-1]["stack_power"]

    assert np.all(stack_power >= 0.0)
    assert np.all(stack_power <= 20.0)
