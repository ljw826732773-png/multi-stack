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
    first = make_cycle_demand("epa_la92")
    second = make_cycle_demand("epa_la92")

    np.testing.assert_allclose(first, second)


def test_epa_aliases_load_the_same_trace():
    np.testing.assert_allclose(make_cycle_demand("epa_la92"), make_cycle_demand("la92"))
    np.testing.assert_allclose(make_cycle_demand("epa_us06"), make_cycle_demand("us06"))


def test_authoritative_cycles_are_default_benchmarks():
    cycles = available_cycles()

    assert cycles[0] == "epa_la92"
    assert "epa_us06" in cycles
    assert "urban" not in cycles
