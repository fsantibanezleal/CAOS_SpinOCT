"""The dimensional contract.

The failure this file exists to prevent is documented in CAOS_MANAGE as
``reference_anchor_slice_constants_break_baselines``: four published methods were broken in one
afternoon by constants written for one observable's units and then applied to another's. In this
domain the risk is higher than usual, because the same symbol genuinely means four different things
across the literature (see ``docs/theory/02-conventions-and-units.md``).
"""

from __future__ import annotations

import math

import pytest

from spinoct.units import (
    BOHR_MAGNETON_J_PER_T,
    BOLTZMANN_J_PER_K,
    ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T,
    MEV_IN_JOULE,
    UNITS,
    CircuitModel,
    bohr_magnetons_to_j_per_t,
    cost_to_joules,
    joules_to_mev,
    landauer_limit_j,
    mev_to_joules,
    self_check,
    thermal_energy_j,
)


def test_self_check_passes() -> None:
    self_check()


def test_constants_match_their_published_values() -> None:
    """CODATA 2022 and the exact SI definitions, to the precision they are quoted at."""
    assert BOHR_MAGNETON_J_PER_T == pytest.approx(9.2740100657e-24, rel=1e-12)
    assert BOLTZMANN_J_PER_K == 1.380649e-23  # exact by SI definition
    assert ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T == pytest.approx(1.76085962784e11, rel=1e-12)
    assert MEV_IN_JOULE == pytest.approx(1.602176634e-22, rel=1e-12)


def test_landauer_limit_at_room_temperature() -> None:
    """2.87 zJ at 300 K, the number this field quotes as its reference line."""
    assert landauer_limit_j(300.0) == pytest.approx(2.8709e-21, rel=1e-4)
    assert landauer_limit_j(300.0) == pytest.approx(thermal_energy_j(300.0) * math.log(2.0), rel=1e-15)


@pytest.mark.parametrize("value", [1e-6, 0.15, 0.31, 1.9, 3.38, 45.0])
def test_energy_round_trip_is_exact(value: float) -> None:
    assert joules_to_mev(mev_to_joules(value)) == pytest.approx(value, rel=1e-15)


def test_moment_conversion() -> None:
    assert bohr_magnetons_to_j_per_t(1.0) == BOHR_MAGNETON_J_PER_T
    assert bohr_magnetons_to_j_per_t(3.0) == pytest.approx(3.0 * BOHR_MAGNETON_J_PER_T, rel=1e-15)


def test_every_declared_unit_is_a_non_empty_string() -> None:
    assert UNITS
    for name, unit in UNITS.items():
        assert isinstance(name, str) and name
        assert isinstance(unit, str) and unit


# ---------------------------------------------------------------------------- the circuit model


def test_circuit_model_refuses_an_empty_description() -> None:
    """A cost converted to joules through an unnamed circuit is not reportable."""
    with pytest.raises(ValueError, match="description"):
        CircuitModel(joules_per_tesla2_second=1.0, description="")
    with pytest.raises(ValueError, match="description"):
        CircuitModel(joules_per_tesla2_second=1.0, description="   ")


def test_circuit_model_refuses_a_non_positive_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        CircuitModel(joules_per_tesla2_second=0.0, description="a coil")
    with pytest.raises(ValueError, match="positive"):
        CircuitModel(joules_per_tesla2_second=-1.0, description="a coil")


def test_cost_to_joules_refuses_a_bare_number() -> None:
    """The signature is the guard: Phi is in T^2 s and only a stated circuit makes it an energy."""
    with pytest.raises(TypeError, match="CircuitModel"):
        cost_to_joules(1.0, 2.0)  # type: ignore[arg-type]


def test_cost_to_joules_is_linear_in_the_cost() -> None:
    circuit = CircuitModel(
        joules_per_tesla2_second=3.0,
        description="a test circuit, R over c squared equal to three",
    )
    assert cost_to_joules(2.0, circuit) == pytest.approx(6.0, rel=1e-15)
    assert cost_to_joules(4.0, circuit) == pytest.approx(2.0 * cost_to_joules(2.0, circuit), rel=1e-15)
