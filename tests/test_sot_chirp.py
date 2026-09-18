"""The simplified spin-orbit-torque protocol (R04) and its published switching probabilities.

The source (Vlasov et al., Phys. Rev. B 105, 134404, after Eq. 15) reports, at f_max = 1.4 f_r,
T = T0, alpha = 0.1, the ideal coupling ratio and a thermal stability factor of 60, a switching
probability of 0.89 at 0.17 j0, 0.97 at 0.18 j0 and practically one at 0.20 j0. This engine does NOT
reproduce those numbers: its deterministic threshold at those settings is 0.21 j0, and at a
stability factor of 60 it switches 0.9 per cent of copies at 0.17 j0 and 90 per cent at 0.25 j0. The
gap is about a factor of 1.4 in amplitude. What was ruled out, measured 2026-09-18: the sense of
rotation (the other sense never switches), the starting tilt and azimuth (no effect up to 0.25 rad),
the coupling convention (the Gilbert and the explicit forms agree to 0.01 in probability at the
ideal ratio), the chirp tuning (the threshold is lowest at 1.2 to 1.4 f_r, as the source chose),
thermal noise (the stochastic 50 per cent point sits on the deterministic threshold), running the
pulse past T, and a factor of two in the time unit. The tests below pin the engine's own behaviour
so that a change in it is visible, and state the published values so the gap cannot be forgotten.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spinoct.analytic.sot import (
    ChirpedRotatingCurrent,
    characteristic_switching_time,
    ideal_sot_ratio_beta,
    resonant_frequency,
)
from spinoct.control import integrate_llg_sot
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

#: The source's published switching probabilities, not reproduced by this engine.
PUBLISHED = {0.17: 0.89, 0.18: 0.97, 0.20: 1.0}


def make_system() -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=0.1
    )


def couplings(alpha: float) -> tuple[float, float]:
    beta = ideal_sot_ratio_beta(alpha)
    return math.cos(beta), math.sin(beta)


def final_sz(pulse: ChirpedRotatingCurrent, steps: int = 3000) -> float:
    system = pulse.system
    xi_f, xi_d = couplings(system.alpha)
    grid = np.linspace(0.0, pulse.switching_time, steps + 1)
    start = np.array([0.0, 0.0, 1.0])
    no_field = np.zeros((grid.size, 3))
    trajectory = integrate_llg_sot(start, no_field, pulse.current_table(grid), grid, system, xi_f, xi_d)
    return float(trajectory[-1, 2])


def test_the_pulse_is_eq_15() -> None:
    system = make_system()
    pulse = ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=0.2)
    assert pulse.switching_time == pytest.approx(characteristic_switching_time(system))
    assert pulse.f_max == pytest.approx(1.4 * resonant_frequency(system))
    grid = np.linspace(0.0, pulse.switching_time, 101)
    np.testing.assert_allclose(np.linalg.norm(pulse.current_table(grid), axis=1), pulse.amplitude)
    # The frequency falls through zero exactly at the barrier crossing, T / 2.
    assert pulse.frequency(pulse.switching_time / 2) == pytest.approx(0.0, abs=1e-6 * pulse.f_max)
    assert pulse.frequency(pulse.switching_time) == pytest.approx(-pulse.f_max)
    assert pulse.cost() == pytest.approx(pulse.amplitude**2 * pulse.switching_time)


def test_only_the_co_rotating_sense_is_resonant() -> None:
    system = make_system()
    co = ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=0.3)
    counter = ChirpedRotatingCurrent(system, co.switching_time, co.amplitude, co.f_max, sense=-1)
    assert final_sz(co) < -0.5
    assert final_sz(counter) > 0.5


def test_the_engine_threshold_sits_above_the_published_amplitudes() -> None:
    """Pins the non-replication: at the published amplitudes this engine does not switch at all.

    If a later change makes 0.17 j0 switch deterministically, the gap has closed and this test says so;
    the published values are in PUBLISHED above.
    """
    system = make_system()
    for amplitude in PUBLISHED:
        pulse = ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=amplitude)
        assert final_sz(pulse) > 0.9, f"{amplitude} j0 now switches: the published gap may have closed"
    assert final_sz(ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=0.25)) < -0.5


def test_the_current_ensemble_reduces_to_the_deterministic_run_at_zero_temperature() -> None:
    """At zero temperature every copy is the deterministic trajectory; this once crashed on 1 / 0."""
    from spinoct.thermal import sot_switching_success_rate

    system = make_system()
    xi_f, xi_d = couplings(system.alpha)
    pulse = ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=0.3)
    result = sot_switching_success_rate(
        system, pulse.current, pulse.switching_time, 0.0, xi_f, xi_d, n_copies=4, n_steps=3000
    )
    assert result.success_rate == 1.0
    assert result.thermal_stability_factor == math.inf
    assert result.final_sz_mean == pytest.approx(final_sz(pulse), abs=0.05)


def test_thermal_noise_makes_the_threshold_statistical() -> None:
    """Near the threshold, at a stability factor of 60, some copies switch and some do not."""
    from spinoct.thermal import sot_switching_success_rate
    from spinoct.units import BOLTZMANN_J_PER_K

    system = make_system()
    xi_f, xi_d = couplings(system.alpha)
    pulse = ChirpedRotatingCurrent.at_source_settings(system, xi=1.0, amplitude_over_j0=0.22)
    temperature = system.anisotropy_j / (BOLTZMANN_J_PER_K * 60.0)
    result = sot_switching_success_rate(
        system, pulse.current, pulse.switching_time, temperature, xi_f, xi_d, n_copies=300, n_steps=3000
    )
    assert 0.2 < result.success_rate < 0.95
