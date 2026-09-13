"""Reversal modes of a spin chain, and the crossover where nonuniform switching becomes cheaper.

Two ways a chain can reverse from all-up to all-down in time T:

- **Uniform rotation.** Every site follows the same macrospin optimal control path, rotating together.
  Exchange costs nothing because neighbouring spins stay parallel throughout. This is the macrospin
  answer, extended trivially to the chain.
- **Domain-wall sweep.** A reversed domain nucleates at one end and its wall propagates to the other,
  so at any instant the chain is part up, part down, joined by a wall. Neighbouring spins across the
  wall are not parallel, so exchange is paid, but only a few sites move at a time, so the field each
  site needs is smaller.

Which is cheaper depends on the chain length and the exchange strength. For a short, strongly exchanged
chain, uniform rotation wins (the macrospin limit). For a long chain, the wall sweep can win because the
field cost of rotating every site at once grows with the number of sites while the wall only ever moves
a few. Finding that crossover is the open problem the macrospin study leaves (Phys. Rev. B 107, 214448,
2023).

The cost of each mode is computed the same way as everywhere in this package: build the trajectory,
invert the equation of motion to get the field each site needs, and integrate the summed squared field.
The field inversion uses the chain internal field, so exchange enters honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analytic.uniaxial import UniaxialOptimalControl
from ..dynamics.system import MacrospinSystem
from .chain import SpinChain

__all__ = ["ReversalComparison", "compare_reversal_modes", "uniform_cost", "domain_wall_cost"]


def _macrospin_of(chain: SpinChain) -> MacrospinSystem:
    return MacrospinSystem(
        mu=chain.mu, anisotropy_j=chain.anisotropy_j, alpha=chain.alpha, gamma=chain.gamma
    )


def _field_cost_of_trajectory(chain: SpinChain, spins_t: np.ndarray, times: np.ndarray) -> float:
    """The switching cost of a chain trajectory, T^2 s.

    Args:
        chain: the spin chain.
        spins_t: the trajectory, shape ``(steps, N, 3)`` unit moments.
        times: the sample times, s, shape ``(steps,)``.

    Returns:
        The total cost ``integral sum_i |b_i|^2 dt`` in T^2 s.
    """
    spins_t = np.asarray(spins_t, dtype=float)
    velocity = np.gradient(spins_t, times, axis=0, edge_order=2)
    alpha, gamma = chain.alpha, chain.gamma

    squared_per_step = np.empty(times.size)
    for k in range(times.size):
        spins = spins_t[k]
        s_dot = velocity[k]
        field = (
            (alpha / gamma) * s_dot
            + (1.0 / gamma) * np.cross(spins, s_dot)
            - chain.internal_field_transverse(spins)
        )
        squared_per_step[k] = np.sum(field**2)
    return float(np.trapezoid(squared_per_step, times))


def _uniform_trajectory(chain: SpinChain, switching_time: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Every site follows the macrospin optimal control path together."""
    macrospin = _macrospin_of(chain)
    optimal = UniaxialOptimalControl.for_switching_time(macrospin, switching_time)
    times = np.linspace(0.0, switching_time, steps)
    single = optimal.moment(times)  # shape (steps, 3)
    spins_t = np.repeat(single[:, None, :], chain.n_sites, axis=1)
    return spins_t, times


def _domain_wall_trajectory(
    chain: SpinChain, switching_time: float, steps: int, wall_width: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """A reversed domain nucleates at site 0 and its wall sweeps to the far end.

    Each site follows a tanh domain-wall profile in time: site ``i`` flips when the wall, moving at
    constant speed across the chain, reaches it. The wall has a finite width so neighbouring spins are
    not discontinuous, which is what keeps the exchange cost finite.

    Args:
        chain: the spin chain.
        switching_time: ``T`` in s.
        steps: the number of time samples.
        wall_width: the wall width in units of the site spacing.

    Returns:
        ``(spins_t, times)`` with ``spins_t`` of shape ``(steps, N, 3)``.
    """
    times = np.linspace(0.0, switching_time, steps)
    n = chain.n_sites
    sites = np.arange(n)
    # The wall centre moves from before site 0 to past site N-1 over the window.
    centre = -2.0 * wall_width + (n - 1 + 4.0 * wall_width) * (times / switching_time)
    spins_t = np.empty((steps, n, 3))
    for k in range(steps):
        # theta goes from 0 (up) to pi (down) as the wall passes; a small azimuthal twist through the
        # wall keeps the reversal a rotation rather than passing through the origin.
        argument = (sites - centre[k]) / wall_width
        s_z = np.tanh(argument)  # +1 ahead of the wall (up), -1 behind it (down)
        theta = np.arccos(np.clip(s_z, -1.0, 1.0))
        phi = 0.5 * np.pi * (1.0 - s_z)  # a gentle in-wall twist
        spins_t[k, :, 0] = np.sin(theta) * np.cos(phi)
        spins_t[k, :, 1] = np.sin(theta) * np.sin(phi)
        spins_t[k, :, 2] = np.cos(theta)
    # Clamp exact endpoints so the boundary conditions are met.
    spins_t[0, :, :] = np.array([0.0, 0.0, 1.0])
    spins_t[-1, :, :] = np.array([0.0, 0.0, -1.0])
    return spins_t, times


def uniform_cost(chain: SpinChain, switching_time: float, steps: int = 2000) -> float:
    """The switching cost of the uniform-rotation reversal, T^2 s."""
    spins_t, times = _uniform_trajectory(chain, switching_time, steps)
    return _field_cost_of_trajectory(chain, spins_t, times)


def domain_wall_cost(
    chain: SpinChain, switching_time: float, steps: int = 2000, wall_width: float = 1.5
) -> float:
    """The switching cost of the domain-wall-sweep reversal, T^2 s."""
    spins_t, times = _domain_wall_trajectory(chain, switching_time, steps, wall_width)
    return _field_cost_of_trajectory(chain, spins_t, times)


@dataclass(frozen=True)
class ReversalComparison:
    """The comparison of reversal modes for one chain and switching time.

    Attributes:
        n_sites: the chain length.
        exchange_over_anisotropy: ``J / K``, the exchange relative to the anisotropy.
        uniform_cost: the uniform-rotation cost, T^2 s.
        domain_wall_cost: the domain-wall-sweep cost, T^2 s.
        cheaper_mode: which mode costs less, "uniform" or "domain-wall".
        ratio: ``domain_wall_cost / uniform_cost``; below one means the wall sweep is cheaper.
    """

    n_sites: int
    exchange_over_anisotropy: float
    uniform_cost: float
    domain_wall_cost: float
    cheaper_mode: str
    ratio: float


def compare_reversal_modes(
    chain: SpinChain, switching_time: float, steps: int = 2000
) -> ReversalComparison:
    """Compare the uniform and domain-wall reversal costs for a chain.

    Args:
        chain: the spin chain.
        switching_time: ``T`` in s.
        steps: the time resolution.

    Returns:
        The :class:`ReversalComparison`.
    """
    uniform = uniform_cost(chain, switching_time, steps)
    wall = domain_wall_cost(chain, switching_time, steps)
    cheaper = "uniform" if uniform <= wall else "domain-wall"
    return ReversalComparison(
        n_sites=chain.n_sites,
        exchange_over_anisotropy=chain.exchange_j / chain.anisotropy_j,
        uniform_cost=uniform,
        domain_wall_cost=wall,
        cheaper_mode=cheaper,
        ratio=wall / uniform if uniform > 0 else float("inf"),
    )
