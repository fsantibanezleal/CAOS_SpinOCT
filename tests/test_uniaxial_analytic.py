"""The exact uniaxial optimal control path, checked against every identity it must satisfy.

These are the package's positive controls. Every numerical solver added later is validated against
them before it is allowed to produce a published number, because a solver that agrees with itself
proves nothing (see the CAOS_MANAGE note ``reference_seed_before_construction_and_test_the_family``).

The identities come from Kwiatkowski, Badarneh, Berkov and Bessarab, Phys. Rev. Lett. 126, 177206
(2021), https://doi.org/10.1103/PhysRevLett.126.177206, and each is checked here by an independent
route: quadrature of the sampled pulse, or forward integration of the equation of motion, rather
than by restating the closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spinoct.analytic import (
    UniaxialOptimalControl,
    cost_free_macrospin,
    cost_infinite_time,
    optimal_switching_time,
)
from spinoct.dynamics import MacrospinSystem, field_from_trajectory
from spinoct.dynamics.llg import integrate_llg, switching_cost
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

DAMPINGS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
SWITCHING_TIMES_IN_TAU0 = [0.5, 2.0, 5.0, 14.0, 50.0]


def make_system(alpha: float) -> MacrospinSystem:
    """A uniaxial macrospin with parameters in the range of a two-dimensional van der Waals magnet.

    A Cr(III) moment of 3 Bohr magnetons and a 0.15 meV easy-axis anisotropy, which puts ``tau0`` in
    the sub-picosecond range where every published result in this field lives.
    """
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0),
        anisotropy_j=mev_to_joules(0.15),
        alpha=alpha,
        hard_axis_ratio=0.0,
        label="test-uniaxial",
    )


def solution(alpha: float, switching_time_in_tau0: float) -> UniaxialOptimalControl:
    system = make_system(alpha)
    return UniaxialOptimalControl.for_switching_time(
        system, system.switching_time_from_tau0(switching_time_in_tau0)
    )


# ---------------------------------------------------------------------------- scales


def test_tau0_is_sub_picosecond_for_realistic_parameters() -> None:
    """A tau0 outside this window means the inputs were handed over in the wrong units."""
    system = make_system(0.1)
    assert 1e-14 < system.tau0 < 1e-11


def test_energy_barrier_is_k_regardless_of_hard_axis() -> None:
    """The biaxial construction changes the switching cost without changing thermal stability."""
    for xi in (0.0, 1.0, 5.0, 10.0):
        system = MacrospinSystem(
            mu=bohr_magnetons_to_j_per_t(3.0),
            anisotropy_j=mev_to_joules(0.15),
            alpha=0.1,
            hard_axis_ratio=xi,
        )
        minimum = float(system.energy(np.array([0.0, 0.0, 1.0])))
        saddle = float(system.energy(np.array([0.0, 1.0, 0.0])))
        assert saddle - minimum == pytest.approx(system.anisotropy_j, rel=1e-14)
        assert system.energy_barrier == pytest.approx(system.anisotropy_j, rel=1e-14)


# ---------------------------------------------------------------------------- boundary conditions


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_path_reverses_the_moment_exactly(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    assert sol.polar_angle(0.0) == pytest.approx(0.0, abs=1e-12)
    assert sol.polar_angle(sol.switching_time) == pytest.approx(math.pi, abs=1e-9)
    assert float(sol.moment(0.0)[2]) == pytest.approx(1.0, abs=1e-12)
    assert float(sol.moment(sol.switching_time)[2]) == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_polar_angle_is_monotone(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    theta = sol.polar_angle(np.linspace(0.0, sol.switching_time, 501))
    assert np.all(np.diff(theta) > 0.0)


# ---------------------------------------------------------------------------- identity 1, symmetry


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_pulse_symmetry_b0_equals_bhalf_equals_bend(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    t_end = sol.switching_time
    values = [
        float(sol.field_amplitude(0.0)),
        float(sol.field_amplitude(0.5 * t_end)),
        float(sol.field_amplitude(t_end)),
    ]
    assert values[1] == pytest.approx(values[0], rel=1e-9)
    assert values[2] == pytest.approx(values[0], rel=1e-9)


# ---------------------------------------------------------------------------- identity 2, spread


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_amplitude_spread_matches_the_closed_form(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 200001)
    sampled = sol.field_amplitude(grid)
    measured_spread = float(sampled.max() - sampled.min())
    assert measured_spread == pytest.approx(sol.amplitude_spread(), rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("alpha", [a for a in DAMPINGS if a > 0.0])
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_extrema_sit_at_quarter_and_three_quarter_points(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 200001)
    sampled = sol.field_amplitude(grid)
    t_max, t_min = sol.peak_times()
    assert grid[int(np.argmax(sampled))] == pytest.approx(t_max, rel=1e-3)
    assert grid[int(np.argmin(sampled))] == pytest.approx(t_min, rel=1e-3)


# ---------------------------------------------------------------------------- identity 3, mean


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_mean_amplitude_is_potential_independent(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 200001)
    measured = float(np.trapezoid(sol.field_amplitude(grid), grid) / sol.switching_time)
    assert measured == pytest.approx(sol.mean_amplitude(), rel=1e-8)


# ---------------------------------------------------------------------------- identity 4, the cost


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_closed_form_cost_matches_quadrature_of_the_pulse(alpha: float, t_tau0: float) -> None:
    """The published cost formula, rederived by integrating the square of the published pulse."""
    sol = solution(alpha, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 400001)
    measured = switching_cost(grid, sol.field_amplitude(grid))
    assert measured == pytest.approx(sol.cost(), rel=1e-8)


# ---------------------------------------------------------------------------- the zero-damping limit


@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_zero_damping_gives_a_constant_pulse_of_pi_over_gamma_t(t_tau0: float) -> None:
    sol = solution(0.0, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 5001)
    expected = math.pi / (sol.system.gamma * sol.switching_time)
    np.testing.assert_allclose(sol.field_amplitude(grid), expected, rtol=1e-12)
    assert sol.amplitude_spread() == pytest.approx(0.0, abs=1e-30)


@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_zero_damping_cost_equals_the_free_macrospin_cost(t_tau0: float) -> None:
    sol = solution(0.0, t_tau0)
    free = cost_free_macrospin(sol.switching_time, 0.0, sol.system.gamma)
    assert sol.cost() == pytest.approx(free, rel=1e-12)
    assert sol.cost_ratio_to_free() == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------- the inequalities


@pytest.mark.parametrize("alpha", DAMPINGS)
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_easy_axis_anisotropy_can_never_help(alpha: float, t_tau0: float) -> None:
    """``Phi_m >= Phi_f`` for a uniaxial system, with equality only at zero damping."""
    sol = solution(alpha, t_tau0)
    assert sol.cost_ratio_to_free() >= 1.0 - 1e-12


@pytest.mark.parametrize("alpha", [a for a in DAMPINGS if a > 0.0])
@pytest.mark.parametrize("t_tau0", SWITCHING_TIMES_IN_TAU0)
def test_cost_never_falls_below_the_universal_floor(alpha: float, t_tau0: float) -> None:
    sol = solution(alpha, t_tau0)
    assert sol.cost() >= cost_infinite_time(sol.system) * (1.0 - 1e-12)


@pytest.mark.parametrize("alpha", [a for a in DAMPINGS if a > 0.0])
def test_cost_decreases_monotonically_with_switching_time(alpha: float) -> None:
    system = make_system(alpha)
    times = system.tau0 * np.array([0.5, 1.0, 2.0, 5.0, 14.0, 50.0, 200.0])
    costs = [UniaxialOptimalControl.for_switching_time(system, t).cost() for t in times]
    assert np.all(np.diff(costs) < 0.0)


# ---------------------------------------------------------------------------- the asymptotics


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.4])
def test_long_time_limit_approaches_the_universal_floor(alpha: float) -> None:
    system = make_system(alpha)
    floor = cost_infinite_time(system)
    # At 40 (alpha + 1/alpha) tau0 the closed form puts the cost within about 1e-8 of the floor,
    # which is still comfortably inside double precision. Going far beyond that is not a stronger
    # test, it is an unrepresentable one (see the dedicated refusal test below).
    long_time = system.tau0 * 40.0 * (alpha + 1.0 / alpha)
    cost = UniaxialOptimalControl.for_switching_time(system, long_time).cost()
    assert cost == pytest.approx(floor, rel=1e-6)
    assert cost >= floor  # the floor is a lower bound, always approached from above


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.4])
def test_switching_time_beyond_double_precision_is_refused(alpha: float) -> None:
    """The floor is approached exponentially, so a very long T saturates below machine epsilon.

    The solver must refuse rather than return a saturated, information-free root. This is the
    representability limit made explicit, not a bug to be worked around.
    """
    from spinoct.analytic import SwitchingTimeTooLongError

    system = make_system(alpha)
    with pytest.raises(SwitchingTimeTooLongError, match="infinite-time floor"):
        UniaxialOptimalControl.for_switching_time(system, system.tau0 * 4000.0 * (alpha + 1.0 / alpha))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.4])
def test_long_time_asymptote_has_the_published_exponential_form(alpha: float) -> None:
    """``Phi_m ~ Phi_inf (1 + 4 exp[-alpha T / (2 tau0 (1+a^2))])`` for ``T >> (a + 1/a) tau0``."""
    system = make_system(alpha)
    floor = cost_infinite_time(system)
    switching_time = system.tau0 * 60.0 * (alpha + 1.0 / alpha)
    cost = UniaxialOptimalControl.for_switching_time(system, switching_time).cost()
    predicted = floor * (
        1.0 + 4.0 * math.exp(-alpha * switching_time / (2.0 * system.tau0 * (1.0 + alpha**2)))
    )
    assert cost == pytest.approx(predicted, rel=5e-3)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_short_time_limit_approaches_the_free_macrospin_cost(alpha: float) -> None:
    """``Phi_m -> pi^2 (1+a^2) / (gamma^2 T)`` for ``T << (a + 1/a) tau0``."""
    system = make_system(alpha)
    switching_time = system.tau0 * 1e-3 * (alpha + 1.0 / alpha)
    cost = UniaxialOptimalControl.for_switching_time(system, switching_time).cost()
    free = cost_free_macrospin(switching_time, alpha, system.gamma)
    assert cost == pytest.approx(free, rel=1e-4)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.4])
@pytest.mark.parametrize("tolerance", [0.05, 0.1, 0.2])
def test_optimal_switching_time_hits_its_defining_tolerance(alpha: float, tolerance: float) -> None:
    """``T_eps`` is defined by ``Phi_m(T_eps) / Phi_inf = 1 + eps``; check the definition holds."""
    system = make_system(alpha)
    t_eps = optimal_switching_time(system, tolerance=tolerance)
    ratio = UniaxialOptimalControl.for_switching_time(system, t_eps).cost_ratio_to_floor()
    assert ratio == pytest.approx(1.0 + tolerance, rel=0.05)


def test_optimal_switching_time_refuses_zero_damping() -> None:
    with pytest.raises(ValueError, match="zero damping"):
        optimal_switching_time(make_system(0.0))


# ---------------------------------------------------------------------------- the physics round trip


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.4])
@pytest.mark.parametrize("t_tau0", [2.0, 14.0])
def test_forward_integration_under_the_optimal_pulse_actually_reverses_the_moment(
    alpha: float, t_tau0: float
) -> None:
    """The decisive check: feed the analytic pulse to the equation of motion and watch it switch.

    Everything above tests the closed forms against each other and against quadrature. This one
    tests them against the physics, by integrating the Landau-Lifshitz-Gilbert equation forward
    under the prescribed field and confirming the moment ends up where the derivation says it does.
    A parametrization spanning zero and finite damping, short and long switching times, is enough to
    exercise the physics; more points here only slow the pure-Python integrator without covering a
    new regime.
    """
    sol = solution(alpha, t_tau0)
    grid = np.linspace(0.0, sol.switching_time, 8001)
    field_table = sol.field_vector(grid)

    def field(t: float) -> np.ndarray:
        return np.array(
            [np.interp(t, grid, field_table[:, k]) for k in range(3)],
            dtype=float,
        )

    trajectory = integrate_llg(np.array([0.0, 0.0, 1.0]), field, grid, sol.system)
    assert trajectory[-1][2] == pytest.approx(-1.0, abs=3e-3)
    np.testing.assert_allclose(np.linalg.norm(trajectory, axis=1), 1.0, atol=1e-12)

    analytic = sol.moment(grid)
    assert float(np.max(np.abs(trajectory[:, 2] - analytic[:, 2]))) < 6e-3


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.4])
@pytest.mark.parametrize("t_tau0", [2.0, 14.0])
def test_inverting_the_equation_of_motion_recovers_the_optimal_field(
    alpha: float, t_tau0: float
) -> None:
    """``field_from_trajectory`` applied to the optimal path must give back the optimal pulse.

    The inversion is an exact algebraic identity given the exact velocity, so the only error here is
    the finite-difference approximation of ``ds/dt``, which is second order in the grid spacing. The
    two grid resolutions below confirm that: refining the grid tenfold shrinks the residual by about
    a hundredfold, which is the signature of a correct inversion sampled with a finite difference,
    not of a systematic error in the code.
    """
    sol = solution(alpha, t_tau0)
    interior = slice(200, -200)

    residuals = []
    for points in (10001, 100001):
        grid = np.linspace(0.0, sol.switching_time, points)
        path = sol.moment(grid)
        velocity = np.gradient(path, grid, axis=0, edge_order=2)
        recovered = field_from_trajectory(path, velocity, sol.system)
        expected = sol.field_vector(grid)
        scale = float(np.max(np.abs(expected)))
        residuals.append(float(np.max(np.abs(recovered[interior] - expected[interior]))) / scale)

    coarse, fine = residuals
    assert fine < 1e-4  # the fine grid recovers the pulse to four digits
    assert fine < coarse / 20.0  # and refining converges at the second-order finite-difference rate


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.4])
def test_the_optimal_field_is_always_perpendicular_to_the_moment(alpha: float) -> None:
    """This perpendicularity is why a longitudinal stabilizing field is invisible to the dynamics."""
    sol = solution(alpha, 5.0)
    grid = np.linspace(0.0, sol.switching_time, 2001)
    path = sol.moment(grid)
    field = sol.field_vector(grid)
    projection = np.sum(path * field, axis=-1)
    assert float(np.max(np.abs(projection))) < 1e-9 * float(np.max(np.abs(field)))


# ---------------------------------------------------------------------------- guards


def test_biaxial_system_is_refused_by_the_closed_form() -> None:
    system = MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0),
        anisotropy_j=mev_to_joules(0.15),
        alpha=0.1,
        hard_axis_ratio=2.0,
    )
    with pytest.raises(ValueError, match="uniaxial"):
        UniaxialOptimalControl.for_switching_time(system, system.tau0)


def test_shape_parameter_reproduces_the_period_relation() -> None:
    from spinoct.analytic.elliptic import complete_k

    for alpha in DAMPINGS:
        system = make_system(alpha)
        for multiple in SWITCHING_TIMES_IN_TAU0:
            switching_time = system.switching_time_from_tau0(multiple)
            sol = UniaxialOptimalControl.for_switching_time(system, switching_time)
            reconstructed = (
                4.0
                * system.tau0
                * (1.0 + alpha**2)
                * sol.p
                * float(complete_k(-(alpha**2) * sol.p**2))
            )
            assert reconstructed == pytest.approx(switching_time, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.1, 0.3])
@pytest.mark.parametrize("t_tau0", [0.5, 5.0, 50.0])
def test_peak_amplitude_is_the_real_peak_not_the_endpoints(alpha: float, t_tau0: float) -> None:
    """The trap this method exists for: the amplitude at the start of the pulse and at its midpoint is
    one and the same middle value, below the peak a quarter of the way through. A driver sized on it
    would be under-specified by up to 27 per cent.

    Checked the independent way: against a dense scan of the pulse rather than by restating the
    closed form.
    """
    system = make_system(alpha)
    switching_time = system.switching_time_from_tau0(t_tau0)
    optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)

    scanned = float(np.max(np.abs(optimal.field_amplitude(np.linspace(0.0, switching_time, 200_001)))))
    assert optimal.peak_amplitude() == pytest.approx(scanned, rel=1e-12)

    ends = max(abs(float(optimal.field_amplitude(t))) for t in (0.0, switching_time / 2.0))
    assert optimal.peak_amplitude() >= ends
    if alpha >= 0.1 and t_tau0 >= 5.0:
        # Where it matters, the difference is not a rounding detail.
        assert optimal.peak_amplitude() > 1.05 * ends


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3])
@pytest.mark.parametrize("t_tau0", [1.0, 20.0])
def test_the_pulse_peaks_at_a_quarter_and_dips_at_three_quarters(alpha: float, t_tau0: float) -> None:
    """Where the extrema of the amplitude are, measured on the pulse itself. The docstring of
    peak_amplitude() in 0.19.000 put the minimum at the start and the midpoint, where the amplitude
    takes a middle value; peak_times() had it right. This pins the shape both rest on."""
    system = make_system(alpha)
    switching_time = system.switching_time_from_tau0(t_tau0)
    optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)
    times = np.linspace(0.0, switching_time, 200_001)
    amplitude = optimal.field_amplitude(times)
    t_max, t_min = optimal.peak_times()
    assert times[int(np.argmax(amplitude))] == pytest.approx(t_max, abs=1e-3 * switching_time)
    assert times[int(np.argmin(amplitude))] == pytest.approx(t_min, abs=1e-3 * switching_time)
    ends_and_middle = (0.0, switching_time / 2.0, switching_time)
    start, middle, end = (float(optimal.field_amplitude(t)) for t in ends_and_middle)
    assert middle == pytest.approx(start, rel=1e-9)
    assert end == pytest.approx(start, rel=1e-9)
    assert amplitude.min() < start < amplitude.max()
