"""The conventional baselines and the analytic SOT protocol.

The baselines must physically reverse the moment when driven hard enough, and cost more than the
optimal control pulse for the same switching. The SOT closed forms are positive controls checked
against their defining relations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spinoct.analytic import SOTOptimalControl, UniaxialOptimalControl, ideal_sot_ratio_beta
from spinoct.control import (
    ConstantFieldProtocol,
    PrecessionalProtocol,
    static_switching_field,
    sun_wang_minimal_field,
)
from spinoct.dynamics import MacrospinSystem
from spinoct.metrics import ProtocolMetrics
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0),
        anisotropy_j=mev_to_joules(0.15),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------- static field


def test_static_field_reverses_when_above_the_switching_field() -> None:
    system = make_system(0.2)
    switching_field = static_switching_field(system)
    switching_time = system.switching_time_from_tau0(200.0)  # slow, damping-limited
    protocol = ConstantFieldProtocol(system, amplitude=1.5 * switching_field)
    result = protocol.run(switching_time)
    assert result.switched
    assert result.final_sz < -0.8


def test_weak_static_field_does_not_reverse() -> None:
    system = make_system(0.2)
    switching_field = static_switching_field(system)
    switching_time = system.switching_time_from_tau0(200.0)
    protocol = ConstantFieldProtocol(system, amplitude=0.1 * switching_field)
    result = protocol.run(switching_time)
    assert not result.switched


def test_sun_wang_minimal_field_is_half_the_anisotropy_field() -> None:
    system = make_system(0.1)
    assert sun_wang_minimal_field(system) == pytest.approx(0.5 * system.anisotropy_field, rel=1e-12)
    assert sun_wang_minimal_field(system) < static_switching_field(system)


# ---------------------------------------------------------------------------- precessional


def test_precessional_pulse_reverses_for_a_well_chosen_duration_and_is_duration_sensitive() -> None:
    """Precessional switching works, but only when the pulse duration is tuned.

    That duration sensitivity is exactly why it is a baseline rather than an optimum: sweeping the
    fraction of the window the transverse field is on, some durations reverse the moment cleanly and
    others do not. A protocol that reversed for every duration would not need optimal control at all.
    """
    system = make_system(0.02)  # low damping, where precession is clean
    switching_time = system.switching_time_from_tau0(10.0)
    amplitude = 2.0 * system.anisotropy_field

    outcomes = [
        PrecessionalProtocol(system, amplitude=amplitude, pulse_fraction=fraction)
        .run(switching_time, n_steps=4001)
        .final_sz
        for fraction in np.linspace(0.1, 0.9, 17)
    ]

    assert min(outcomes) < -0.8  # at least one duration reverses cleanly
    assert max(outcomes) > 0.0  # and at least one does not, so the outcome depends on the duration


# ---------------------------------------------------------------------------- baselines cost more


def test_a_conventional_pulse_costs_more_than_the_optimal_pulse() -> None:
    """The point of optimal control: for the same switching, the optimal pulse is cheaper."""
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(50.0)

    optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)

    # A static field strong enough to reverse in this window.
    static = ConstantFieldProtocol(system, amplitude=1.2 * static_switching_field(system))
    static_result = static.run(switching_time)
    assert static_result.switched
    assert static_result.cost > optimal.cost()


# ---------------------------------------------------------------------------- SOT closed forms


def test_ideal_sot_ratio_beta_makes_the_torque_purely_switching() -> None:
    """At beta*, tan(beta* + eta) = 0, the ideal ratio xi_D = -alpha xi_F."""
    for alpha in (0.05, 0.1, 0.2, 0.4):
        beta_star = ideal_sot_ratio_beta(alpha)
        eta = math.atan(alpha)
        assert math.tan(beta_star + eta) == pytest.approx(0.0, abs=1e-12)


def test_sot_forbidden_ratio_is_detected() -> None:
    """xi_F = alpha xi_D, that is tan(beta) = alpha, prohibits switching (the cost diverges)."""
    system = make_system(0.2)
    beta_forbidden = math.atan(system.alpha)
    protocol = SOTOptimalControl(system, switching_time=system.tau0 * 5.0, xi=1.0, beta=beta_forbidden)
    assert protocol.is_forbidden()

    beta_ideal = ideal_sot_ratio_beta(system.alpha)
    ideal = SOTOptimalControl(system, switching_time=system.tau0 * 5.0, xi=1.0, beta=beta_ideal)
    assert not ideal.is_forbidden()


def test_sot_mean_current_matches_its_closed_form() -> None:
    from spinoct.analytic.elliptic import complete_k

    system = make_system(0.1)
    switching_time = system.tau0 * 20.0
    beta = 0.3
    protocol = SOTOptimalControl(system, switching_time=switching_time, xi=1.5, beta=beta)
    eta = math.atan(system.alpha)
    j0 = system.anisotropy_j / (system.mu * 1.5)
    expected = (
        4.0
        * j0
        * math.sqrt(1.0 + system.alpha**2)
        * float(complete_k(math.sin(beta + eta) ** 2))
        / switching_time
    )
    assert protocol.mean_current() == pytest.approx(expected, rel=1e-12)


def test_sot_characteristic_time_matches_its_closed_form() -> None:
    system = make_system(0.2)
    protocol = SOTOptimalControl(
        system, switching_time=system.tau0 * 10.0, xi=1.0, beta=ideal_sot_ratio_beta(0.2)
    )
    expected = (1.0 + 0.2**2) * math.pi**2 * system.tau0 / (2.0 * 0.2)
    assert protocol.characteristic_time_ideal() == pytest.approx(expected, rel=1e-12)


def test_sot_ideal_mean_current_scales_linearly_in_damping() -> None:
    """The low-damping advantage: <j*> = 4 alpha j0 / (pi sqrt(1+a^2)), linear in alpha."""
    system = make_system(0.05)
    protocol = SOTOptimalControl(
        system, switching_time=system.tau0 * 10.0, xi=1.0, beta=ideal_sot_ratio_beta(0.05)
    )
    j0 = system.anisotropy_j / (system.mu * 1.0)
    expected = 4.0 * 0.05 * j0 / (math.pi * math.sqrt(1.0 + 0.05**2))
    assert protocol.mean_current_ideal() == pytest.approx(expected, rel=1e-12)


def test_sot_rejects_biaxial_and_nonpositive_coupling() -> None:
    biaxial = MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=0.1, hard_axis_ratio=2.0
    )
    with pytest.raises(ValueError, match="uniaxial"):
        SOTOptimalControl(biaxial, switching_time=1e-12, xi=1.0, beta=0.0)
    with pytest.raises(ValueError, match="xi"):
        SOTOptimalControl(make_system(0.1), switching_time=1e-12, xi=0.0, beta=0.0)


# ---------------------------------------------------------------------------- metrics


def test_metrics_score_a_protocol_consistently() -> None:
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(50.0)
    static = ConstantFieldProtocol(system, amplitude=1.2 * static_switching_field(system))
    result = static.run(switching_time)
    metrics = ProtocolMetrics.score(
        system, switching_time, result.times, result.field, result.cost, result.final_sz
    )
    assert metrics.cost == result.cost
    assert metrics.cost_over_floor > 0.0
    assert metrics.peak_amplitude == pytest.approx(
        float(np.max(np.linalg.norm(result.field, axis=-1))), rel=1e-12
    )
    assert metrics.switched == result.switched
    assert set(metrics.as_dict()) == {
        "cost",
        "cost_over_floor",
        "cost_over_free",
        "peak_amplitude",
        "bandwidth_hz",
        "switched",
        "final_sz",
    }
