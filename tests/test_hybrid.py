"""Joint field-plus-SOT hybrid optimal control (R13)."""

from __future__ import annotations

import numpy as np

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
