"""The chain minimum energy path and the barrier floor it puts under the switching cost.

Positive controls: one uniaxial site has barrier K and its floor is the analytic infinite-time
optimum; a short chain reverses coherently (barrier N K); a long chain's barrier saturates at the
domain-wall energy, which the continuum theory puts at 2 sqrt(2 J K) for this Hamiltonian. And the
floor must lie under every optimal cost the package computes.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl
from spinoct.analytic.uniaxial import cost_infinite_time
from spinoct.dynamics import MacrospinSystem
from spinoct.lattice import LatticeOCPSolver, SpinChain, cost_floor_from_barrier, minimum_energy_path
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

MU = bohr_magnetons_to_j_per_t(3.0)
K = mev_to_joules(0.15)
ALPHA = 0.1


def chain(n_sites: int, exchange_over_k: float) -> SpinChain:
    return SpinChain(n_sites=n_sites, mu=MU, anisotropy_j=K, exchange_j=exchange_over_k * K, alpha=ALPHA)


def test_single_site_barrier_is_k_and_its_floor_is_the_infinite_time_optimum() -> None:
    single = chain(1, 0.0)
    path = minimum_energy_path(single, n_images=21, initial="uniform")
    assert path.converged
    assert path.barrier == pytest.approx(K, rel=1e-6)
    floor = cost_floor_from_barrier(path.barrier, MU, ALPHA, single.gamma)
    macrospin = MacrospinSystem(mu=MU, anisotropy_j=K, alpha=ALPHA)
    assert floor == pytest.approx(cost_infinite_time(macrospin), rel=1e-6)


def test_short_chain_reverses_coherently() -> None:
    short = chain(4, 10.0)
    path = minimum_energy_path(short, initial="wall")
    assert path.converged
    assert path.barrier_over_uniform(short) == pytest.approx(1.0, abs=1e-4)


def test_long_chain_barrier_saturates_near_the_wall_energy() -> None:
    exchange_over_k = 10.0
    barriers = []
    for n_sites in (16, 24, 32):
        path = minimum_energy_path(chain(n_sites, exchange_over_k), initial="wall")
        assert path.converged
        barriers.append(path.barrier / K)
    continuum_wall = 2.0 * np.sqrt(2.0 * exchange_over_k)
    assert barriers[-1] == pytest.approx(continuum_wall, rel=0.02)
    assert abs(barriers[-1] - barriers[-2]) < 1e-2
    assert barriers[0] <= barriers[-1] + 1e-9


def test_saddle_image_is_the_energy_maximum_and_endpoints_are_the_poles() -> None:
    path = minimum_energy_path(chain(12, 10.0), initial="wall")
    np.testing.assert_allclose(path.images[0, :, 2], 1.0)
    np.testing.assert_allclose(path.images[-1, :, 2], -1.0)
    assert path.saddle_index == int(np.argmax(path.energies))
    assert path.energies[0] == 0.0


@pytest.mark.parametrize("switching_tau0", [4.0, 20.0])
def test_floor_lies_under_every_optimal_cost(switching_tau0: float) -> None:
    n_sites = 6
    c = chain(n_sites, 2.0)
    floor = cost_floor_from_barrier(minimum_energy_path(c).barrier, MU, ALPHA, c.gamma)
    macrospin = MacrospinSystem(mu=MU, anisotropy_j=K, alpha=ALPHA)
    switching_time = switching_tau0 * c.tau0
    analytic_uniform = n_sites * UniaxialOptimalControl.for_switching_time(macrospin, switching_time).cost()
    assert floor <= analytic_uniform
    free = LatticeOCPSolver(c, 40, switching_time).solve(initial="wall", noise=0.0, max_iterations=400)
    assert floor <= free.cost


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        minimum_energy_path(chain(3, 1.0), n_images=2)
    with pytest.raises(ValueError):
        minimum_energy_path(chain(3, 1.0), initial="spiral")


def test_wide_walls_converge_onto_the_continuum_wall_energy() -> None:
    """The string method must converge for wide walls, and the barrier must approach 2 sqrt(2 J K).

    A fixed explicit step crossed the stability limit just above J / K = 24; the path oscillated, hit the
    iteration cap, and still returned a barrier 1.5 times the continuum value at J / K = 25 and 43 times it
    at J / K = 40. The deficit below the continuum must also shrink like one over the wall width squared,
    the leading discreteness correction, which is what an exact lattice result converging on a continuum
    closed form looks like.
    """
    import math

    from spinoct.lattice import SpinChain, minimum_energy_path
    from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

    mu, anisotropy = bohr_magnetons_to_j_per_t(3.0), mev_to_joules(0.15)
    widths, deficits = [], []
    for exchange_over_k in (10.0, 20.0, 40.0):
        width_squared = exchange_over_k / 2.0
        n_sites = max(24, int(12 * math.sqrt(width_squared)))
        exchange = exchange_over_k * anisotropy
        chain = SpinChain(n_sites=n_sites, mu=mu, anisotropy_j=anisotropy, exchange_j=exchange, alpha=0.1)
        path = minimum_energy_path(chain, initial="wall")
        assert path.converged, f"J/K = {exchange_over_k} did not converge"
        continuum = 2.0 * math.sqrt(2.0 * exchange_over_k) * anisotropy
        ratio = path.barrier / continuum
        assert 0.99 < ratio < 1.0, f"J/K = {exchange_over_k}: barrier/continuum = {ratio}"
        widths.append(width_squared)
        deficits.append(1.0 - ratio)
    # Deficit ~ c / w^2: the log-log slope over a factor of four in w^2 is -1 within ten per cent.
    slope = math.log(deficits[-1] / deficits[0]) / math.log(widths[-1] / widths[0])
    assert -1.1 < slope < -0.9, f"discreteness deficit scales as w^{2 * slope:.2f}, not w^-2"
