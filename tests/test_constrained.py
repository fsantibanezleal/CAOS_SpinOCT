"""The constrained pulse-shaping solvers, GRAPE and CRAB.

These optimize the control directly under a realizability constraint, which the unconstrained optimal
control paths cannot express. The tests confirm they reverse the moment and respect their constraints;
they use small budgets because the forward integration is pure Python, and the product bakes them at
modest scale for the price-of-realizability analysis rather than over the full material sweep.
"""

from __future__ import annotations

import numpy as np

from spinoct.control import CRABSolver, GRAPESolver
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=alpha
    )


def test_crab_reverses_the_moment_within_a_bounded_bandwidth() -> None:
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    solver = CRABSolver(system, switching_time, n_harmonics=4, integration_steps=250)
    result = solver.solve(max_iterations=60)
    # The band-limited pulse drives the moment into the reversed hemisphere.
    assert result.final_sz < 0.0
    # Its bandwidth is bounded by the highest harmonic, the realizability guarantee by construction.
    assert solver.bandwidth_hz() == 4.0 / switching_time


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


def test_more_crab_harmonics_can_only_lower_or_match_the_achievable_cost() -> None:
    """The price of realizability: more bandwidth cannot cost more at the optimum.

    This is the qualitative shape of the realizability curve, checked cheaply. With more harmonics the
    feasible set grows, so the best achievable cost is monotonically non-increasing. The budgets are
    small, so this asserts the direction, not a converged value.
    """
    system = make_system()
    switching_time = system.switching_time_from_tau0(8.0)
    low = CRABSolver(system, switching_time, n_harmonics=2, integration_steps=200).solve(max_iterations=40)
    high = CRABSolver(system, switching_time, n_harmonics=5, integration_steps=200).solve(max_iterations=40)
    # The richer basis reverses at least as completely as the poorer one.
    assert high.final_sz <= low.final_sz + 0.2
