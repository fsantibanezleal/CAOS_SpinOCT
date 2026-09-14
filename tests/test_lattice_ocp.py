"""The free chain optimal control path (Gap 1, full).

The solver minimizes the switching cost over every site's trajectory. It is trusted only through
properties it cannot satisfy by accident:

1. A one-site chain with no exchange is a macrospin, so its discrete optimum must match the analytic
   cost up to the midpoint-rule discretization error.
2. The colored central-difference gradient must equal a naive per-image central difference.
3. Uniform rotation is feasible and exchange-free, so the reported bound must equal ``N`` times the
   single-site cost exactly, and a solve started there can only go down.
4. The wall seed must be a valid trajectory: unit vectors, clamped poles, every site reversing.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl
from spinoct.dynamics import MacrospinSystem
from spinoct.lattice import LatticeOCPSolver, SpinChain
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

MU = bohr_magnetons_to_j_per_t(3.0)
K = mev_to_joules(0.15)
ALPHA = 0.1


def chain(n_sites: int, exchange_over_k: float) -> SpinChain:
    return SpinChain(n_sites=n_sites, mu=MU, anisotropy_j=K, exchange_j=exchange_over_k * K, alpha=ALPHA)


def test_single_site_optimum_matches_the_analytic_cost() -> None:
    single = chain(1, 0.0)
    switching_time = 4.0 * single.tau0
    solver = LatticeOCPSolver(single, 60, switching_time)
    result = solver.solve(initial="uniform", noise=0.0)
    analytic = UniaxialOptimalControl.for_switching_time(
        MacrospinSystem(mu=MU, anisotropy_j=K, alpha=ALPHA), switching_time
    ).cost()
    assert result.cost == pytest.approx(analytic, rel=5e-3)
    assert result.nonuniformity == 0.0


def test_colored_gradient_equals_naive_central_difference() -> None:
    small = chain(5, 2.0)
    solver = LatticeOCPSolver(small, 6, 3.0 * small.tau0)
    rng = np.random.default_rng(3)
    images = solver._wall_path()
    images[1:-1] += rng.normal(scale=0.1, size=images[1:-1].shape)
    images[1:-1] /= np.linalg.norm(images[1:-1], axis=-1, keepdims=True)

    colored = solver._gradient(images)

    step = 1e-7
    naive = np.zeros_like(images)
    for q in range(1, images.shape[0] - 1):
        for i in range(images.shape[1]):
            for c in range(3):
                plus = images.copy()
                plus[q, i, c] += step
                minus = images.copy()
                minus[q, i, c] -= step
                naive[q, i, c] = (solver.cost_of(plus) - solver.cost_of(minus)) / (2.0 * step)
    naive -= np.sum(naive * images, axis=-1, keepdims=True) * images
    naive[0] = 0.0
    naive[-1] = 0.0

    scale = np.max(np.abs(naive))
    assert np.max(np.abs(colored - naive)) <= 1e-5 * scale


def test_uniform_bound_is_n_times_the_single_site_cost() -> None:
    n_sites = 6
    solver = LatticeOCPSolver(chain(n_sites, 5.0), 20, 4.0 * chain(1, 0.0).tau0)
    single = LatticeOCPSolver(chain(1, 0.0), 20, 4.0 * chain(1, 0.0).tau0)
    assert solver.uniform_bound() == pytest.approx(n_sites * single.uniform_bound(), rel=1e-12)


def test_solve_from_the_uniform_optimum_never_exceeds_the_bound() -> None:
    solver = LatticeOCPSolver(chain(6, 1.0), 20, 4.0 * chain(1, 0.0).tau0)
    result = solver.solve(initial="uniform", noise=0.0, max_iterations=200)
    assert result.cost <= result.uniform_bound * (1.0 + 1e-12)
    assert result.saving_vs_uniform >= -1e-12


def test_wall_seed_is_a_valid_reversing_trajectory() -> None:
    solver = LatticeOCPSolver(chain(10, 8.0), 30, 20.0 * chain(1, 0.0).tau0)
    path = solver._wall_path()
    assert path.shape == (32, 10, 3)
    np.testing.assert_allclose(np.linalg.norm(path, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(path[0, :, 2], 1.0)
    np.testing.assert_allclose(path[-1, :, 2], -1.0)
    # The wall enters at site 0: the first site is always at least as reversed as the last.
    assert np.all(path[1:-1, 0, 2] <= path[1:-1, -1, 2] + 1e-12)
    assert solver.wall_width_sites() == pytest.approx(2.0)


def test_wall_seed_never_jumps_far_between_consecutive_images() -> None:
    """A seed that compresses the precession makes near-antipodal jumps, where the midpoint rule is
    singular; that trap pinned a strong-exchange solve at 51 times the bound."""
    solver = LatticeOCPSolver(chain(8, 25.0), 80, 40.0 * chain(1, 0.0).tau0)
    path = solver._wall_path()
    step_angles = np.arccos(np.clip(np.sum(path[:-1] * path[1:], axis=-1), -1.0, 1.0))
    assert step_angles.max() < 0.5
    # The fixed seed costs about 200 times the bound (a fast wall); the compressed seed cost 12000.
    assert solver.cost_of(path) < 1000.0 * solver.uniform_bound()


def test_mep_seed_is_a_valid_smooth_reversing_trajectory() -> None:
    solver = LatticeOCPSolver(chain(12, 10.0), 60, 30.0 * chain(1, 0.0).tau0)
    path = solver._mep_path()
    np.testing.assert_allclose(np.linalg.norm(path, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(path[0, :, 2], 1.0)
    np.testing.assert_allclose(path[-1, :, 2], -1.0)
    step_angles = np.arccos(np.clip(np.sum(path[:-1] * path[1:], axis=-1), -1.0, 1.0))
    assert step_angles.max() < 0.8


def test_long_chain_at_long_time_reverses_more_cheaply_through_a_wall() -> None:
    """The headline: above the barrier crossover length and at long switching time, the free optimum is
    nonuniform and strictly cheaper than uniform rotation. Every reported cost is that of an explicit
    feasible trajectory on the same grid as the bound, so this is an upper bound on the true optimum
    even when the optimizer is stopped early."""
    long_chain = SpinChain(n_sites=10, mu=MU, anisotropy_j=K, exchange_j=6.0 * K, alpha=0.5)
    solver = LatticeOCPSolver(long_chain, 60, 40.0 * long_chain.tau0)
    result = solver.solve(initial="wall", noise=0.0, max_iterations=400)
    assert result.cost < 0.9 * result.uniform_bound
    assert result.nonuniformity > 0.3


def test_short_chain_stays_uniform() -> None:
    short = SpinChain(n_sites=4, mu=MU, anisotropy_j=K, exchange_j=6.0 * K, alpha=0.5)
    solver = LatticeOCPSolver(short, 60, 40.0 * short.tau0)
    result = solver.solve(initial="wall", noise=0.0, max_iterations=400)
    assert result.cost == pytest.approx(result.uniform_bound, rel=2e-3)


def test_recommended_images_grow_with_switching_time_and_length() -> None:
    base = LatticeOCPSolver.recommended_images(chain(8, 10.0), 20.0 * chain(1, 0.0).tau0)
    longer_time = LatticeOCPSolver.recommended_images(chain(8, 10.0), 150.0 * chain(1, 0.0).tau0)
    longer_chain = LatticeOCPSolver.recommended_images(chain(200, 10.0), 20.0 * chain(1, 0.0).tau0)
    assert base >= 40
    assert longer_time > base
    assert longer_chain > base


def test_solve_is_reproducible_for_a_seed() -> None:
    solver = LatticeOCPSolver(chain(4, 1.0), 12, 4.0 * chain(1, 0.0).tau0)
    first = solver.solve(initial="wall", seed=7, noise=0.05, max_iterations=50)
    second = solver.solve(initial="wall", seed=7, noise=0.05, max_iterations=50)
    np.testing.assert_array_equal(first.images, second.images)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        LatticeOCPSolver(chain(2, 1.0), 0, 1e-12)
    with pytest.raises(ValueError):
        LatticeOCPSolver(chain(2, 1.0), 5, 0.0)
    solver = LatticeOCPSolver(chain(2, 1.0), 5, 4.0 * chain(1, 0.0).tau0)
    with pytest.raises(ValueError):
        solver.solve(initial="spiral")
    assert solver.solve(initial="mep", max_iterations=2).images.shape == (7, 2, 3)
    with pytest.raises(ValueError):
        solver.solve(method="newton")
