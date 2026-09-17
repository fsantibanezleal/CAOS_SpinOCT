"""The numerical image-based optimal control path solver, gated against the analytic solution.

A numerical solver is trusted on problems with no closed form only after it reproduces the ones that
do have a closed form. So the decisive tests here compare the numerical optimal control path against
`spinoct.analytic.uniaxial` on the uniaxial macrospin, where the exact answer is known.

The solves use modest image counts so the suite stays fast; the reference uses up to 1500 images for
publication accuracy, but a few tens already demonstrate correctness and second-order convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl, cost_free_macrospin
from spinoct.dynamics import MacrospinSystem
from spinoct.numeric import ImageOCPSolver
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system(alpha: float, hard_axis_ratio: float = 0.0) -> MacrospinSystem:
    return MacrospinSystem(
        mu=bohr_magnetons_to_j_per_t(3.0),
        anisotropy_j=mev_to_joules(0.15),
        alpha=alpha,
        hard_axis_ratio=hard_axis_ratio,
    )


# ---------------------------------------------------------------------------- the acceptance gate


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_numeric_cost_matches_the_analytic_uniaxial_cost(alpha: float) -> None:
    """The keystone gate: the numerical solver reproduces the exact uniaxial cost."""
    system = make_system(alpha)
    switching_time = system.switching_time_from_tau0(3.0)
    analytic = UniaxialOptimalControl.for_switching_time(system, switching_time)

    result = ImageOCPSolver(system, n_images=80, switching_time=switching_time).solve(
        max_iterations=4000
    )
    ratio = result.cost / analytic.cost()
    # The midpoint discretization is above the exact cost and converges to it from above; 80 images
    # puts it within a fraction of a percent.
    assert 1.0 <= ratio < 1.01


def test_discretization_error_falls_as_images_are_added() -> None:
    """Refining the chain must reduce the gap to the analytic cost, the signature of convergence."""
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(3.0)
    exact = UniaxialOptimalControl.for_switching_time(system, switching_time).cost()

    errors = []
    for n_images in (30, 120):
        result = ImageOCPSolver(system, n_images=n_images, switching_time=switching_time).solve(
            max_iterations=5000
        )
        errors.append(result.cost / exact - 1.0)

    coarse, fine = errors
    assert fine < coarse
    assert fine < 0.005


# ---------------------------------------------------------------------------- the physics


def test_the_uniaxial_path_precesses_rather_than_following_a_flat_meridian() -> None:
    """The optimal control path precesses; the pole-to-pole meridian is only a saddle.

    A solver that reported the meridian would pass a cost check that was loose enough but be
    physically wrong. The right invariant is the peak-to-peak swing of the azimuth, not its net
    change: by the axial symmetry the moment advances in azimuth up to the equator and returns, so
    the net change is near zero while the peak-to-peak swing is large (the analytic value for this
    case is about 1.84 radians). A flat meridian has a peak-to-peak swing of zero.
    """
    system = make_system(0.2)
    switching_time = system.switching_time_from_tau0(6.0)
    analytic = UniaxialOptimalControl.for_switching_time(system, switching_time)
    analytic_azimuth = analytic.azimuthal_angle(np.linspace(0.0, switching_time, 2001))
    analytic_swing = float(analytic_azimuth.max() - analytic_azimuth.min())

    result = ImageOCPSolver(system, n_images=120, switching_time=switching_time).solve(
        max_iterations=12000, relative_tolerance=1e-14
    )
    interior = result.images[3:-3]
    azimuth = np.unwrap(np.arctan2(interior[:, 1], interior[:, 0]))
    numeric_swing = float(azimuth.max() - azimuth.min())

    assert analytic_swing > 1.5  # the analytic path genuinely precesses in this regime
    assert numeric_swing > 0.8 * analytic_swing  # and the numeric path recovers most of that swing


def test_hard_axis_pushes_the_cost_below_the_free_macrospin_floor() -> None:
    """The central scientific claim of the biaxial mechanism, checked numerically.

    A uniaxial magnet can never beat the free-macrospin cost, but a hard axis can, because its
    internal torque assists the reversal in part of configuration space. There is no closed form
    here, so this is where the numerical solver earns its keep, and the multi-seed sweep guards
    against reporting a symmetric local minimum instead of the true optimum.
    """
    switching_time = make_system(0.1).switching_time_from_tau0(2.0)
    free = cost_free_macrospin(switching_time, 0.1, make_system(0.1).gamma)

    uniaxial = ImageOCPSolver(make_system(0.1, 0.0), n_images=60, switching_time=switching_time)
    biaxial = ImageOCPSolver(make_system(0.1, 5.0), n_images=60, switching_time=switching_time)

    uniaxial_cost = uniaxial.solve_best(n_seeds=3, max_iterations=2500).cost
    biaxial_cost = biaxial.solve_best(n_seeds=4, max_iterations=2500).cost

    assert uniaxial_cost >= 0.99 * free  # uniaxial cannot beat the free floor
    assert biaxial_cost < 0.6 * free  # the hard axis buys a large reduction


# ---------------------------------------------------------------------------- reproducibility


def test_the_same_seed_gives_the_same_result() -> None:
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(3.0)
    solver = ImageOCPSolver(system, n_images=40, switching_time=switching_time)
    first = solver.solve(seed=7, max_iterations=1500)
    second = solver.solve(seed=7, max_iterations=1500)
    assert first.cost == second.cost
    np.testing.assert_array_equal(first.images, second.images)


def test_solve_best_is_reproducible_and_no_worse_than_any_single_seed() -> None:
    system = make_system(0.1, 3.0)
    switching_time = system.switching_time_from_tau0(2.0)
    solver = ImageOCPSolver(system, n_images=40, switching_time=switching_time)

    best = solver.solve_best(n_seeds=3, base_seed=0, max_iterations=2000)
    singles = [
        solver.solve(seed=seed, noise=0.15, max_iterations=2000).cost for seed in range(3)
    ]
    assert best.cost <= min(singles) + 1e-30


# ---------------------------------------------------------------------------- structure


def test_endpoints_are_clamped_and_the_chain_stays_on_the_sphere() -> None:
    system = make_system(0.1)
    switching_time = system.switching_time_from_tau0(3.0)
    result = ImageOCPSolver(system, n_images=50, switching_time=switching_time).solve(
        max_iterations=800
    )
    np.testing.assert_allclose(result.images[0], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.images[-1], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result.images, axis=1), 1.0, atol=1e-10)
    # Q = 50 interior images gives 52 images, 51 intervals, so 51 midpoint fields.
    assert result.field_midpoints.shape == (51, 3)
    assert result.times.shape == (52,)


def test_solver_rejects_bad_construction() -> None:
    system = make_system(0.1)
    with pytest.raises(ValueError, match="n_images"):
        ImageOCPSolver(system, n_images=0, switching_time=1e-12)
    with pytest.raises(ValueError, match="positive"):
        ImageOCPSolver(system, n_images=10, switching_time=0.0)
    with pytest.raises(ValueError, match="n_seeds"):
        ImageOCPSolver(system, n_images=10, switching_time=1e-12).solve_best(n_seeds=0)


def test_recommended_images_grow_with_switching_time_and_fix_the_long_time_error() -> None:
    """A fixed image count degrades as the path spirals: the rule must scale, and the cost must improve.

    Measured before the rule existed: 60 images put the numerical cost 20 per cent above the closed form
    at T = 100 tau0, while matching it at T = 2 tau0.
    """
    system = make_system(alpha=0.01)
    short = system.switching_time_from_tau0(2.0)
    long = system.switching_time_from_tau0(100.0)
    assert ImageOCPSolver.recommended_images(system, short) == 60
    assert ImageOCPSolver.recommended_images(system, long) > 3 * 60

    analytic = UniaxialOptimalControl.for_switching_time(system, long).cost()
    fixed = ImageOCPSolver(system, 60, long).solve_best(n_seeds=2, max_iterations=2000).cost
    scaled = ImageOCPSolver(
        system, ImageOCPSolver.recommended_images(system, long), long
    ).solve_best(n_seeds=2, max_iterations=2000).cost
    assert fixed / analytic > 1.05, "the regression this rule exists for"
    assert scaled / analytic < fixed / analytic
