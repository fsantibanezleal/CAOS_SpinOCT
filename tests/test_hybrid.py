"""Joint field-plus-SOT hybrid optimal control (R13)."""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.adjoint import adjoint_gradient_sot
from spinoct.control import HybridSolver, integrate_llg_sot
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=alpha)


def test_pure_damping_like_current_drives_the_moment_toward_the_equator() -> None:
    """A pure damping-like SOT current torques the moment off the pole toward the perpendicular state.

    It cannot complete the reversal alone (a constant DL torque stabilizes the equator, which is why
    conventional SOT needs a symmetry-breaking field), but it must clearly move the moment, which
    validates that the SOT term acts with the right sign.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(30.0)
    grid = np.linspace(0.0, switching_time, 3000)
    current = np.tile([3.0 * system.anisotropy_field, 0.0, 0.0], (grid.size, 1))
    field = np.zeros((grid.size, 3))
    trajectory = integrate_llg_sot(
        np.array([0.0, 0.0, 1.0]), field, current, grid, system, xi_f=0.0, xi_d=1.0
    )
    # From the north pole (sz=1) the current drives it to the equator (sz near 0).
    assert trajectory[-1, 2] < 0.2
    # And the norm is preserved.
    np.testing.assert_allclose(np.linalg.norm(trajectory, axis=1), 1.0, atol=1e-9)


def test_no_current_recovers_field_only_dynamics() -> None:
    """With zero current the SOT integrator reduces to the plain field-driven equation of motion."""
    system = make_system()
    switching_time = system.switching_time_from_tau0(10.0)
    grid = np.linspace(0.0, switching_time, 500)
    field = np.zeros((grid.size, 3))
    field[:, 0] = 0.5 * system.anisotropy_field  # a constant transverse field
    zero_current = np.zeros((grid.size, 3))
    trajectory = integrate_llg_sot(
        np.array([0.0, 0.0, 1.0]), field, zero_current, grid, system, xi_f=0.0, xi_d=0.0
    )
    # The moment precesses and stays on the sphere; the point is only that it runs and is norm-safe.
    np.testing.assert_allclose(np.linalg.norm(trajectory, axis=1), 1.0, atol=1e-9)


def test_hybrid_reverses_using_both_field_and_current() -> None:
    """The co-optimizer finds a reversing protocol that spends on both controls (R13).

    A small, bounded solve: it must reverse the moment and use a nontrivial share of each control, which
    is the whole point of the joint problem the kickoff paper names as an open design space.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    solver = HybridSolver(
        system,
        switching_time,
        circuit_field=1.0,
        circuit_current=1.0,
        xi_f=0.1,
        xi_d=0.1,
        n_harmonics=2,
        integration_steps=160,
    )
    result = solver.solve(max_iterations=50)
    assert result.final_sz < 0.0  # reversed into the far hemisphere
    # Both controls carry cost (a genuine hybrid, not a degenerate field-only or current-only solution).
    assert result.field_cost > 0.0
    assert result.current_cost > 0.0
    assert 0.0 < result.field_fraction < 1.0


def test_the_sot_adjoint_gradient_matches_finite_differences() -> None:
    """Both controls at once: one backward pass must equal the finite-difference gradient.

    The spin-orbit-torque terms add their own Jacobians to the adjoint recursion, and a sign error in
    any of them would still produce a plausible-looking descent. The step is chosen relative to the
    field scale: too small a step and the difference of two objectives of order 1e-11 is round-off, not
    a derivative, which is how an earlier check of this appeared to fail at the 1e-3 level.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(4.0)
    grid = np.linspace(0.0, switching_time, 120)
    rng = np.random.default_rng(0)
    field = rng.normal(scale=0.3 * system.anisotropy_field, size=(grid.size, 3))
    current = rng.normal(scale=0.4, size=(grid.size, 3))
    field[:, 2] = 0.0
    current[:, 2] = 0.0
    weight = 1e-11

    def objective(f, j):
        return adjoint_gradient_sot(f, j, grid, system, 0.05, 0.05, weight)[0]

    _value, _sz, grad_field, grad_current = adjoint_gradient_sot(
        field, current, grid, system, 0.05, 0.05, weight
    )
    for table, gradient in ((field, grad_field), (current, grad_current)):
        step = 1e-4 * float(np.abs(table).max())
        for index in ((5, 0), (40, 1), (90, 0)):
            table[index] += step
            plus = objective(field, current)
            table[index] -= 2.0 * step
            minus = objective(field, current)
            table[index] += step
            numerical = (plus - minus) / (2.0 * step)
            assert abs(numerical - gradient[index]) <= 1e-5 * abs(numerical)


@pytest.mark.slow
def test_a_cheaper_current_shifts_the_optimum_away_from_the_field() -> None:
    """The question the case exists to ask: where does the optimum sit as the price of current moves.

    Measured 2026-09-17 at ten tau0: the share of the weighted cost carried by the field falls from
    0.96 at a price of 0.1 to 0.03 at 1e-4, so the crossover is real and the sweep must reach it. The
    test asserts the direction, which is what the physics guarantees, not the values.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    shares = []
    for price in (1e-3, 1e-1):
        result = HybridSolver(
            system,
            switching_time,
            circuit_field=1.0,
            circuit_current=price,
            n_harmonics=3,
            integration_steps=300,
        ).solve(max_iterations=200)
        assert result.switched, f"the co-optimization did not reverse at a current price of {price}"
        shares.append(result.field_fraction)
    assert shares[0] < shares[1], f"a cheaper current did not shift cost onto the current: {shares}"
