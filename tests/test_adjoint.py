"""Exact gradient-based pulse optimization by the discrete adjoint method (R10).

The decisive test of an adjoint gradient is that it matches a finite-difference gradient of the same
objective; the decisive test of the optimizer is that it converges onto the analytic optimum.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.adjoint import adjoint_gradient, optimize_pulse_adjoint
from spinoct.analytic import UniaxialOptimalControl
from spinoct.dynamics import MacrospinSystem
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=alpha)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_adjoint_gradient_matches_finite_differences(alpha: float) -> None:
    """The hand-derived reverse-mode gradient is exact for the discretized system."""
    system = make_system(alpha)
    switching_time = system.switching_time_from_tau0(5.0)
    n = 24
    times = np.linspace(0.0, switching_time, n)
    rng = np.random.default_rng(1)
    field = rng.normal(scale=0.5 * system.anisotropy_field, size=(n, 3))
    field[:, 2] = 0.0
    weight = 1e-10

    _, _, grad = adjoint_gradient(field, times, system, weight)
    step = 1e-3 * system.anisotropy_field
    for k, c in [(3, 0), (7, 1), (12, 0), (18, 1)]:
        fp = field.copy()
        fp[k, c] += step
        fm = field.copy()
        fm[k, c] -= step
        op = adjoint_gradient(fp, times, system, weight)[0]
        om = adjoint_gradient(fm, times, system, weight)[0]
        fd = (op - om) / (2.0 * step)
        assert grad[k, c] == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("t_tau0", [5.0, 8.0])
def test_adjoint_optimizer_converges_onto_the_analytic_optimum(t_tau0: float) -> None:
    """The exact gradient fed to L-BFGS finds a reversal at nearly the closed-form optimal cost."""
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(t_tau0)
    analytic = UniaxialOptimalControl.for_switching_time(system, switching_time).cost()

    result = optimize_pulse_adjoint(system, switching_time, n_slices=60, max_iterations=400)
    assert result.switched
    assert result.final_sz < -0.99
    # The piecewise-constant discretization lands within a few percent of the exact optimal cost.
    assert 0.9 < result.cost / analytic < 1.1


def test_adjoint_optimizer_is_reproducible() -> None:
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(6.0)
    a = optimize_pulse_adjoint(system, switching_time, n_slices=40, max_iterations=200, seed=3)
    b = optimize_pulse_adjoint(system, switching_time, n_slices=40, max_iterations=200, seed=3)
    assert a.cost == b.cost
    np.testing.assert_array_equal(a.field, b.field)
