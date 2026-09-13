"""The spin chain: optimal control beyond the macrospin (Gap 1).

The lattice solver is validated the only honest way, against the macrospin it must reduce to: a
one-site chain with no exchange reproduces the analytic macrospin cost. Then the reversal-mode
comparison is checked for its scaling, which is the physics of the crossover question the macrospin
study leaves open.
"""

from __future__ import annotations

import numpy as np
import pytest

from spinoct.analytic import UniaxialOptimalControl
from spinoct.dynamics import MacrospinSystem
from spinoct.lattice import SpinChain, compare_reversal_modes, domain_wall_cost, uniform_cost
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules

MU = bohr_magnetons_to_j_per_t(3.0)
K = mev_to_joules(0.15)


def macrospin(alpha: float = 0.1) -> MacrospinSystem:
    return MacrospinSystem(mu=MU, anisotropy_j=K, alpha=alpha)


# ---------------------------------------------------------------------------- the validation limit


def test_single_site_chain_reproduces_the_macrospin_cost() -> None:
    """A one-site chain with no exchange IS a macrospin; the costs must agree."""
    chain = SpinChain(n_sites=1, mu=MU, anisotropy_j=K, exchange_j=0.0, alpha=0.1)
    mac = macrospin()
    switching_time = mac.switching_time_from_tau0(10.0)
    analytic = UniaxialOptimalControl.for_switching_time(mac, switching_time).cost()
    numeric = uniform_cost(chain, switching_time, steps=4000)
    assert numeric == pytest.approx(analytic, rel=2e-3)


def test_uniform_reversal_costs_scale_linearly_with_chain_length() -> None:
    """In the uniform mode exchange is never paid, so N sites cost N times one site."""
    switching_time = SpinChain(1, MU, K, 0.0, 0.1).tau0 * 10.0
    costs = []
    for n in (2, 4, 8, 16):
        chain = SpinChain(n_sites=n, mu=MU, anisotropy_j=K, exchange_j=0.5 * K, alpha=0.1)
        costs.append(uniform_cost(chain, switching_time, steps=1000))
    costs = np.array(costs)
    # Each doubling of the chain doubles the cost.
    ratios = costs[1:] / costs[:-1]
    np.testing.assert_allclose(ratios, 2.0, rtol=0.02)


# ---------------------------------------------------------------------------- the internal field


def test_exchange_field_couples_neighbours() -> None:
    chain = SpinChain(n_sites=3, mu=MU, anisotropy_j=K, exchange_j=K, alpha=0.1)
    # All up: the exchange field on the middle site points along z from both neighbours.
    up = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    field = chain.internal_field(up)
    # Middle site has two neighbours, so twice the exchange field of an end site's exchange part.
    exchange_field = chain.exchange_field
    assert field[1, 2] == pytest.approx(2.0 * K / MU + 2.0 * exchange_field, rel=1e-12)
    assert field[0, 2] == pytest.approx(2.0 * K / MU + exchange_field, rel=1e-12)


# ---------------------------------------------------------------------------- the Gap-1 result


def test_uniform_rotation_is_the_field_cost_optimum_over_a_domain_wall_sweep() -> None:
    """The honest Gap-1 finding for the Joule-heating cost metric.

    A domain-wall sweep, the mode that dominates real thermally-driven switching, is more expensive in
    switching COST than uniform rotation, because the wall forces fast local flips and pays exchange
    across itself, while uniform rotation moves every site slowly and never bends a bond. This holds
    across the chain lengths and exchange strengths tested. A full free lattice optimal control path
    (future work) could find a cheaper nonuniform mode; this first comparison finds uniform optimal.
    """
    switching_time = SpinChain(1, MU, K, 0.0, 0.1).tau0 * 10.0
    for n in (8, 32):
        for exchange_ratio in (0.5, 2.0):
            chain = SpinChain(
                n_sites=n, mu=MU, anisotropy_j=K, exchange_j=exchange_ratio * K, alpha=0.1
            )
            comparison = compare_reversal_modes(chain, switching_time, steps=1000)
            assert comparison.cheaper_mode == "uniform"
            assert comparison.domain_wall_cost > comparison.uniform_cost


def test_domain_wall_cost_grows_at_least_as_fast_as_uniform_with_length() -> None:
    """The wall does not become the field-cost optimum by lengthening the chain in this ansatz."""
    switching_time = SpinChain(1, MU, K, 0.0, 0.1).tau0 * 10.0
    short = SpinChain(8, MU, K, 0.5 * K, 0.1)
    long = SpinChain(64, MU, K, 0.5 * K, 0.1)
    short_ratio = domain_wall_cost(short, switching_time, steps=800) / uniform_cost(
        short, switching_time, steps=800
    )
    long_ratio = domain_wall_cost(long, switching_time, steps=800) / uniform_cost(
        long, switching_time, steps=800
    )
    assert long_ratio >= short_ratio
