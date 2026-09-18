"""The two-dimensional square patch: its energy, its colouring, and the solvers running on it.

A patch is a new geometry for solvers written first for a chain, so the tests check the three places a
geometry enters: the physics (the field is minus the energy gradient), the gradient machinery (the
five-colour gradient equals a brute-force one), and the solvers' answers (a straight wall across a
patch costs exactly its height times the chain barrier). The chain itself must be untouched by the
generalization, which the chain suites and the bit-identical comparison in the changelog cover.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.lattice import LatticeOCPSolver, SpinChain, SpinPatch, minimum_energy_path
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

MU = bohr_magnetons_to_j_per_t(3.0)
K = mev_to_joules(0.15)


def make_patch(width: int, height: int, exchange_over_k: float = 10.0, alpha: float = 0.3) -> SpinPatch:
    exchange = exchange_over_k * K
    return SpinPatch(width=width, height=height, mu=MU, anisotropy_j=K, exchange_j=exchange, alpha=alpha)


def unit(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    v = rng.normal(size=shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def closed_neighbourhood(patch: SpinPatch, i: int) -> set[int]:
    x, y = i % patch.width, i // patch.width
    return {
        j
        for j in range(patch.n_sites)
        if abs(j % patch.width - x) + abs(j // patch.width - y) <= 1
    }


def test_the_field_is_minus_the_energy_gradient() -> None:
    patch = make_patch(5, 4, exchange_over_k=3.0)
    spins = unit(np.random.default_rng(0), (patch.n_sites, 3))
    field = patch.internal_field(spins)
    step = 1e-6
    for i in range(patch.n_sites):
        for c in range(3):
            plus, minus = spins.copy(), spins.copy()
            plus[i, c] += step
            minus[i, c] -= step
            numerical = -(patch.energy(plus) - patch.energy(minus)) / (2.0 * step) / patch.mu
            assert numerical == pytest.approx(field[i, c], rel=1e-6, abs=1e-6 * np.abs(field).max())


def test_same_colour_sites_have_disjoint_closed_neighbourhoods() -> None:
    patch = make_patch(9, 7)
    colors = patch.colors()
    for color in range(int(colors.max()) + 1):
        seen: set[int] = set()
        for i in np.flatnonzero(colors == color):
            footprint = closed_neighbourhood(patch, int(i))
            assert not footprint & seen
            seen |= footprint


def test_the_footprint_sum_is_the_closed_neighbourhood_sum() -> None:
    patch = make_patch(6, 5)
    values = np.random.default_rng(1).normal(size=(3, patch.n_sites))
    footprints = [closed_neighbourhood(patch, i) for i in range(patch.n_sites)]
    expected = np.array([[sum(row[j] for j in footprint) for footprint in footprints] for row in values])
    np.testing.assert_allclose(patch.footprint_sum(values), expected, atol=1e-12)


def test_the_five_colour_gradient_equals_a_brute_force_gradient() -> None:
    """60 cost evaluations must give the same gradient as perturbing every coordinate on its own."""
    patch = make_patch(5, 4, exchange_over_k=3.0)
    solver = LatticeOCPSolver(patch, 5, 10.0 * patch.tau0)
    rng = np.random.default_rng(2)
    images = solver._wall_path() + 0.05 * rng.normal(size=(7, patch.n_sites, 3))
    images /= np.linalg.norm(images, axis=-1, keepdims=True)
    images[0], images[-1] = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]

    colored = solver._gradient(images)
    step = 1e-7
    brute = np.zeros_like(images)
    for q in range(1, images.shape[0] - 1):
        for i in range(patch.n_sites):
            for c in range(3):
                plus, minus = images.copy(), images.copy()
                plus[q, i, c] += step
                minus[q, i, c] -= step
                brute[q, i, c] = (solver.cost_of(plus) - solver.cost_of(minus)) / (2.0 * step)
    brute -= np.sum(brute * images, axis=-1, keepdims=True) * images
    brute[0], brute[-1] = 0.0, 0.0
    assert np.abs(colored - brute).max() <= 1e-6 * np.abs(brute).max()


@pytest.mark.parametrize(("width", "height"), [(8, 4), (16, 8)])
def test_a_straight_wall_costs_its_height_times_the_chain_barrier(width: int, height: int) -> None:
    """Rows at the same angle pay no exchange between them, so the barrier is exactly H times the chain's.

    The string method once computed a chain's energy whatever lattice it was given, so a flattened patch
    was treated as one long snake of sites and every patch barrier came out as the saturated chain value.
    """
    chain = SpinChain(n_sites=width, mu=MU, anisotropy_j=K, exchange_j=10.0 * K, alpha=0.3)
    chain_path = minimum_energy_path(chain, initial="wall")
    patch_path = minimum_energy_path(make_patch(width, height), initial="wall")
    assert patch_path.converged
    assert patch_path.barrier == pytest.approx(height * chain_path.barrier, rel=1e-6)


def test_a_small_patch_reverses_uniformly_at_the_uniform_bound() -> None:
    """Below the crossover size the optimum is uniform rotation, and it sits on the exact uniform bound."""
    patch = make_patch(2, 2)
    switching_time = 10.0 * patch.tau0
    images = LatticeOCPSolver.recommended_images(patch, switching_time)
    solver = LatticeOCPSolver(patch, images, switching_time)
    result = solver.solve(initial="uniform", seed=0, noise=0.0, max_iterations=300)
    assert result.cost / solver.uniform_bound() == pytest.approx(1.0, abs=1e-6)
    assert result.nonuniformity < 1e-3
