"""The constrained pulse-shaping solvers, GRAPE and CRAB.

These optimize the control directly under a realizability constraint, which the unconstrained optimal
control paths cannot express. Both are driven by the exact adjoint gradient through a linear control
basis, so the tests here check the two things that actually go wrong:

- the gradient really is the gradient (against finite differences), and
- the answers obey the two inequalities the physics guarantees: more bandwidth cannot cost more, and no
  pulse that reverses the moment can cost less than the analytic optimum.

The second one is the test that was missing. The previous version asserted only that a richer basis
reversed "at least as completely" as a poorer one, with a slack of 0.2 in the final ``s_z``, which is
satisfied by almost anything. Under that test the solver shipped returning 2.2 times the analytic
optimum at two harmonics and 14 times at six, where a strictly larger search space cannot cost more.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl
from spinoct.control import CRABSolver, GRAPESolver
from spinoct.control.linear_basis import LinearControlProblem, harmonic_design, interpolation_design
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=alpha
    )


# ------------------------------------------------------------------ the gradient


def test_the_adjoint_gradient_matches_finite_differences() -> None:
    """The whole method rests on this: one backward pass must equal the finite-difference gradient.

    Checked through the linear basis, so it covers the design-matrix chain rule as well as the adjoint
    itself. A central difference on a smooth objective is accurate to the square of the step, so a
    relative agreement of 1e-6 leaves no room for a sign error or a missing term.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(4.0)
    grid = np.linspace(0.0, switching_time, 220)
    design = harmonic_design(np.array([1.0, 2.0]) / switching_time, grid, switching_time)
    problem = LinearControlProblem(
        system=system,
        times=grid,
        design=design,
        fidelity_weight=1e-11,
        field_scale=system.anisotropy_field,
        objective_scale=1e-11,
    )
    rng = np.random.default_rng(0)
    point = rng.normal(scale=0.8, size=2 * design.shape[1])

    _value, analytic = problem.objective_and_gradient(point)
    step = 1e-6
    numerical = np.empty_like(analytic)
    for i in range(point.size):
        forward, backward = point.copy(), point.copy()
        forward[i] += step
        backward[i] -= step
        numerical[i] = (
            problem.objective_and_gradient(forward)[0] - problem.objective_and_gradient(backward)[0]
        ) / (2.0 * step)

    assert np.allclose(analytic, numerical, rtol=1e-6, atol=1e-8 * np.max(np.abs(numerical)))


def test_the_interpolation_design_reproduces_numpy_interp() -> None:
    """The slice-to-grid map is a matrix; it must be the same map numpy applies."""
    nodes = np.linspace(0.0, 1.0, 6)
    grid = np.linspace(0.0, 1.0, 41)
    values = np.array([0.0, 1.0, -2.0, 0.5, 3.0, -1.0])
    assert np.allclose(interpolation_design(nodes, grid) @ values, np.interp(grid, nodes, values))


def test_the_harmonic_basis_carries_both_quadratures() -> None:
    """Sine only would fix the field direction in the plane; the optimal pulse rotates.

    With one column per harmonic the x and y components share a phase at every frequency, so the drive
    is linear and no number of harmonics can make it rotate. That cost 6.2 times the analytic optimum
    and did not improve with bandwidth.
    """
    grid = np.linspace(0.0, 1.0, 101)
    design = harmonic_design(np.array([1.0, 2.0, 3.0]), grid, 1.0)
    assert design.shape == (101, 6)
    # The sine and cosine columns of one harmonic are orthogonal over the window, so the basis really
    # does span two quadratures rather than repeating one.
    assert abs(float(design[:, 0] @ design[:, 3])) < 1e-9 * float(design[:, 0] @ design[:, 0])


# ------------------------------------------------------------------ the inequalities the physics gives


@pytest.mark.slow
def test_more_crab_harmonics_can_only_lower_or_match_the_cost() -> None:
    """The price of realizability: more bandwidth is a strictly larger feasible set.

    Asserted on the COST, which is the quantity the case reports, with every answer required to be a
    real reversal first. A tolerance of one per cent allows for the optimizer stopping at slightly
    different points, not for the trend reversing.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    costs = []
    for harmonics in (2, 4, 6):
        result = CRABSolver(
            system, switching_time, n_harmonics=harmonics, integration_steps=600
        ).solve(max_iterations=300)
        assert result.switched, f"{harmonics} harmonics did not reverse the moment"
        costs.append(result.cost)
    for lower, higher in pairwise(costs):
        assert higher <= lower * 1.01, f"more bandwidth cost more: {costs}"


@pytest.mark.slow
def test_no_constrained_pulse_beats_the_analytic_optimum() -> None:
    """The floor: the closed form is the minimum cost of a complete reversal at this time.

    A constrained solver can only match it or pay more. A reported cost below it means the pulse did
    not actually finish the reversal and banked the saving, which is what the reversal threshold is
    there to prevent.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    optimum = UniaxialOptimalControl.for_switching_time(system, switching_time).cost()

    loose = GRAPESolver(
        system,
        switching_time,
        n_slices=20,
        amplitude_cap=20.0 * system.anisotropy_field,
        integration_steps=600,
    ).solve(max_iterations=300)
    assert loose.switched
    assert loose.cost >= optimum * 0.995
    # And an unconstrained-in-practice GRAPE run should come close to it: this is the positive control
    # that the direct solver and the closed form describe the same problem.
    assert loose.cost <= optimum * 1.2

    band_limited = CRABSolver(
        system, switching_time, n_harmonics=4, integration_steps=600
    ).solve(max_iterations=300)
    assert band_limited.switched
    assert band_limited.cost >= optimum * 0.995


# ------------------------------------------------------------------ the constraints themselves


def test_grape_respects_the_amplitude_cap() -> None:
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    cap = 5.0 * system.anisotropy_field
    solver = GRAPESolver(
        system, switching_time, n_slices=10, amplitude_cap=cap, integration_steps=250
    )
    result = solver.solve(max_iterations=40)
    # No component ever exceeds the box constraint (a small tolerance for the interpolation).
    assert np.max(np.abs(result.field)) <= cap * 1.0001


@pytest.mark.slow
def test_a_cap_too_small_reports_no_reversal_rather_than_a_cheap_pulse() -> None:
    """Below the amplitude that can reverse the moment in the given time, there is no answer.

    The honest report is a pulse that did not switch, not a cheap cost. Measured 2026-09-17 on this
    system at eight tau0: the threshold sits between 0.4 and 0.6 anisotropy fields per component.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    result = GRAPESolver(
        system,
        switching_time,
        n_slices=20,
        amplitude_cap=0.15 * system.anisotropy_field,
        integration_steps=600,
    ).solve(max_iterations=200)
    assert not result.switched
    assert result.infidelity > 0.1


def test_crab_bandwidth_is_the_highest_harmonic() -> None:
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    solver = CRABSolver(system, switching_time, n_harmonics=4, integration_steps=250)
    assert solver.bandwidth_hz() == 4.0 / switching_time
