"""Thermal dynamics and reliability: the stochastic thermostat, and the longitudinal-field front.

The thermostat is validated against the Boltzmann distribution it must reproduce, and the
longitudinal-stabilization front is validated against the published physics of arXiv:2312.11293: the
half-hyperbolic bare path, the stabilization to unity at large longitudinal field, and the added cost
that field carries.

Ensembles are small so the suite stays fast; the trends are robust to the sample size, and the binomial
confidence interval is reported so a borderline point is legible.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.dynamics import MacrospinSystem
from spinoct.thermal import (
    boltzmann_polar_variance,
    br_cost_reliability_front,
    hyperbolic_fraction,
    instability_penalty,
    perturbation_eigenvalues,
    stochastic_llg_step,
)
from spinoct.thermal.stochastic import thermal_field_scale
from spinoct.units import BOLTZMANN_J_PER_K, bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=alpha
    )


def temperature_for_stability(system: MacrospinSystem, delta: float) -> float:
    """The temperature giving a target thermal stability factor Delta = K / (k_B T)."""
    return system.anisotropy_j / (BOLTZMANN_J_PER_K * delta)


# ---------------------------------------------------------------------------- the thermostat


def test_thermal_field_vanishes_at_zero_temperature() -> None:
    system = make_system()
    assert thermal_field_scale(system, 0.0, 1e-14) == 0.0
    assert thermal_field_scale(system, 100.0, 1e-14) > 0.0


def test_thermostat_reproduces_the_boltzmann_distribution() -> None:
    """The decisive thermostat check: the fluctuation-dissipation noise gives <theta^2> = k_B T / K."""
    system = make_system(alpha=0.1)
    delta = 40.0
    temperature = temperature_for_stability(system, delta)
    rng = np.random.default_rng(0)

    n = 4000
    s = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    dt = 0.02 * system.tau0
    for _ in range(4000):
        s = stochastic_llg_step(s, np.zeros(3), system, temperature, dt, rng)

    theta = np.arccos(np.clip(s[:, 2], -1.0, 1.0))
    measured = float(np.mean(theta**2))
    predicted = boltzmann_polar_variance(system, temperature)
    assert measured == pytest.approx(predicted, rel=0.1)


def test_zero_temperature_step_is_deterministic() -> None:
    system = make_system()
    rng = np.random.default_rng(0)
    s = np.array([[np.sin(0.3), 0.0, np.cos(0.3)]])
    a = stochastic_llg_step(s, np.zeros(3), system, 0.0, 1e-15, rng)
    b = stochastic_llg_step(s, np.zeros(3), system, 0.0, 1e-15, np.random.default_rng(999))
    np.testing.assert_allclose(a, b, atol=1e-14)


# ---------------------------------------------------------------------------- the eigenvalues


def test_perturbation_eigenvalues_match_the_closed_form() -> None:
    system = make_system()
    theta = np.linspace(0.0, np.pi, 11)
    for br in (0.0, 0.5 * system.anisotropy_field, system.anisotropy_field):
        w1, w2 = perturbation_eigenvalues(theta, br, system)
        np.testing.assert_allclose(w1, br + system.anisotropy_field * np.cos(2 * theta), atol=1e-18)
        np.testing.assert_allclose(w2, br + system.anisotropy_field * np.cos(theta) ** 2, atol=1e-18)


def test_bare_path_is_half_hyperbolic_and_a_large_field_removes_it() -> None:
    """At B_r = 0 the reversal is hyperbolic for pi/4 < theta < 3pi/4 (half the path)."""
    system = make_system()
    theta = np.linspace(0.0, np.pi, 2001)
    bare = hyperbolic_fraction(theta, 0.0, system)
    assert bare == pytest.approx(0.5, abs=0.05)
    stabilized = hyperbolic_fraction(theta, 2.0 * system.anisotropy_field, system)
    assert stabilized == 0.0


def test_instability_penalty_is_positive_bare_and_zero_when_stabilized() -> None:
    system = make_system()
    switching_time = system.switching_time_from_tau0(10.0)
    times = np.linspace(0.0, switching_time, 1001)
    theta = np.linspace(0.0, np.pi, 1001)
    assert instability_penalty(theta, times, system, longitudinal_field=0.0) > 0.0
    assert instability_penalty(
        theta, times, system, longitudinal_field=2.0 * system.anisotropy_field
    ) == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------- the front


def test_longitudinal_field_front_improves_reliability_at_a_cost() -> None:
    """The novel R12 result: a large longitudinal field reaches unity success, at a cost it also carries.

    Reproduces the published physics (arXiv:2312.11293): the bare path is only partly reliable, a large
    longitudinal field removes the hyperbolic instability and drives success to near unity, and the
    added cost grows quadratically in the field.
    """
    system = make_system(alpha=0.1)
    switching_time = system.switching_time_from_tau0(10.0)
    temperature = temperature_for_stability(system, 20.0)

    front = br_cost_reliability_front(
        system,
        switching_time,
        temperature,
        br_over_anisotropy=(0.0, 1.0, 2.0),
        n_copies=200,
        n_steps=400,
    )

    bare, mid, strong = front
    # The bare path is dynamically unstable over half its length; a strong field removes that.
    assert bare.hyperbolic_fraction == pytest.approx(0.5, abs=0.1)
    assert strong.hyperbolic_fraction == 0.0
    # The strong field reaches near-unity reliability, clearly better than the bare pulse.
    assert strong.success_rate > bare.success_rate
    assert strong.success_rate > 0.95
    # And it is not free: the added cost grows with the square of the field.
    assert bare.added_cost == 0.0
    assert strong.added_cost > mid.added_cost > 0.0
    assert strong.added_cost == pytest.approx(4.0 * mid.added_cost, rel=1e-9)

def test_the_analysis_and_the_ensemble_agree_on_which_field_stabilises() -> None:
    """The eigenvalues and the simulation in this module must mean the same thing by B_r.

    `perturbation_eigenvalues` calls a positive B_r stabilizing, so the ensemble has to measure a
    HIGHER success rate there. Until 0.18.000 the front added the longitudinal component with the
    opposite sign, so the two contradicted each other: `hyperbolic_fraction` reported a fully stable
    path while the ensemble it was meant to predict got worse. Measured at a stability factor of one
    and alpha = 0.01, the old sign took the success rate from 0.780 down to 0.530 and this one takes
    it up to 0.958.
    """
    system = make_system(alpha=0.01)
    switching_time = system.switching_time_from_tau0(10.0)
    front = br_cost_reliability_front(
        system,
        switching_time,
        temperature_for_stability(system, 1.0),
        br_over_anisotropy=(0.0, 1.0),
        n_copies=400,
        n_steps=600,
    )
    bare, stabilised = front
    assert bare.hyperbolic_fraction > 0.2, "the bare path has an instability to remove"
    # At exactly one anisotropy field an eigenvalue touches zero at theta = pi/2, so a sample or
    # two sit on the boundary; the instability itself is gone.
    assert stabilised.hyperbolic_fraction < 0.01, "and the analysis says this field removes it"
    assert stabilised.success_rate > bare.success_rate + bare.confidence95, "so the ensemble must agree"
