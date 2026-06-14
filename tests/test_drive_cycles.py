import numpy as np

from multistack_ai import available_cycles, make_cycle_demand


def test_available_cycles_have_finite_power_profiles():
    for cycle in available_cycles():
        demand = make_cycle_demand(cycle)

        assert demand.ndim == 1
        assert len(demand) > 100
        assert np.all(np.isfinite(demand))
        assert np.min(demand) >= -120.0
        assert np.max(demand) <= 280.0


def test_cycle_generation_is_deterministic():
    first = make_cycle_demand("urban")
    second = make_cycle_demand("urban")

    np.testing.assert_allclose(first, second)
