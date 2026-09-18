"""A two-dimensional square patch of spins, the geometry of a real memory element.

The energy of a ``W x H`` patch of unit moments is

    E = -K sum_i s_{i,z}^2  -  J sum_{<ij>} s_i . s_j

with the same easy-axis anisotropy ``K`` per site and ferromagnetic nearest-neighbour exchange ``J`` as
the chain, now over the four in-plane neighbours of a square lattice, with open boundaries. A chain is a
line through the element; a patch is the element. The chain result (walls become the optimal reversal
above a crossover length at long switching time) is a statement about one dimension, and whether it
survives in two is the question this geometry exists to ask.

Sites are stored flattened, index ``i = y W + x``, so every array the solvers handle has the same shape
``(..., N, 3)`` as for a chain. The patch implements the same lattice interface as
:class:`~spinoct.lattice.chain.SpinChain`: a batched internal field, a colouring with disjoint closed
neighbourhoods, the footprint sum, a wall coordinate and the coordination.

The colouring. Perturbing one site changes the cost of that site and of its four neighbours, its closed
neighbourhood. On the square lattice the colouring ``(x + 2 y) mod 5`` assigns every site of a closed
neighbourhood a different colour, and two sites of one colour have disjoint closed neighbourhoods (the
five-colour perfect code of the grid). So one colour class can be perturbed at once and every site's
derivative read off its own footprint, and the exact gradient costs 2 x 5 x 3 x 2 = 60 cost evaluations
whatever the patch size, against 36 for a chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T

__all__ = ["SpinPatch"]

#: The spatial colour count of the square-lattice perfect code.
_PATCH_COLORS = 5


@dataclass(frozen=True)
class SpinPatch:
    """A ``width x height`` square patch of unit moments with open boundaries.

    Attributes:
        width: sites along x, the direction a wall travels.
        height: sites along y.
        mu: the magnetic moment per site, J/T.
        anisotropy_j: the easy-axis anisotropy per site ``K``, J.
        exchange_j: the nearest-neighbour exchange ``J``, J. Positive is ferromagnetic.
        alpha: Gilbert damping, dimensionless.
        gamma: gyromagnetic ratio, rad/(s T).
        label: a short display name.
    """

    width: int
    height: int
    mu: float
    anisotropy_j: float
    exchange_j: float
    alpha: float
    gamma: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T
    label: str = "patch"

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("a patch needs at least one site in each direction")
        if self.mu <= 0.0 or self.anisotropy_j <= 0.0:
            raise ValueError("the moment and the easy-axis anisotropy must be positive")
        if self.alpha < 0.0:
            raise ValueError("damping must be non-negative")

    # ------------------------------------------------------------------ geometry

    @property
    def n_sites(self) -> int:
        return self.width * self.height

    @property
    def coordination(self) -> int:
        """The largest number of neighbours of a site: four on a square lattice."""
        return 4

    def _xy(self) -> tuple[np.ndarray, np.ndarray]:
        index = np.arange(self.n_sites)
        return index % self.width, index // self.width

    @property
    def tau0(self) -> float:
        """The single-site Larmor timescale ``mu / (2 gamma K)``, s."""
        return self.mu / (2.0 * self.gamma * self.anisotropy_j)

    @property
    def anisotropy_field(self) -> float:
        return self.anisotropy_j / self.mu

    # ------------------------------------------------------------------ energy and field

    def _grid(self, spins: np.ndarray) -> np.ndarray:
        """View flattened spins ``(..., N, 3)`` as ``(..., H, W, 3)``."""
        return spins.reshape(*spins.shape[:-2], self.height, self.width, 3)

    def energy(self, spins: np.ndarray) -> float:
        """Total energy of one configuration, J."""
        spins = np.asarray(spins, dtype=float)
        grid = self._grid(spins)
        anisotropy = -self.anisotropy_j * np.sum(spins[:, 2] ** 2)
        bonds = np.sum(grid[:, :-1] * grid[:, 1:]) + np.sum(grid[:-1, :] * grid[1:, :])
        return float(anisotropy - self.exchange_j * bonds)

    def internal_field_batched(self, spins: np.ndarray) -> np.ndarray:
        """The internal field for a stack of configurations, shape ``(P, N, 3)``, T."""
        spins = np.asarray(spins, dtype=float)
        out = np.zeros_like(spins)
        out[..., 2] += 2.0 * self.anisotropy_j / self.mu * spins[..., 2]
        coupling = self.exchange_j / self.mu
        grid = self._grid(spins)
        field = self._grid(out)
        field[..., :, :-1, :] += coupling * grid[..., :, 1:, :]
        field[..., :, 1:, :] += coupling * grid[..., :, :-1, :]
        field[..., :-1, :, :] += coupling * grid[..., 1:, :, :]
        field[..., 1:, :, :] += coupling * grid[..., :-1, :, :]
        return out

    def internal_field(self, spins: np.ndarray) -> np.ndarray:
        """The internal field of one configuration, shape ``(N, 3)``, T."""
        return self.internal_field_batched(np.asarray(spins, dtype=float)[None])[0]

    def internal_field_transverse(self, spins: np.ndarray) -> np.ndarray:
        """The per-site internal field with its component along each moment removed."""
        spins = np.asarray(spins, dtype=float)
        field = self.internal_field(spins)
        return field - np.sum(field * spins, axis=-1, keepdims=True) * spins

    # ------------------------------------------------------------------ the lattice interface

    def colors(self) -> np.ndarray:
        """The perfect-code colouring ``(x + 2 y) mod 5``: same-colour closed neighbourhoods are disjoint."""
        x, y = self._xy()
        return (x + 2 * y) % _PATCH_COLORS

    def footprint_sum(self, per_site: np.ndarray) -> np.ndarray:
        """Sum a per-site quantity over each site's closed neighbourhood, along the last axis."""
        grid = per_site.reshape(*per_site.shape[:-1], self.height, self.width)
        padded = np.pad(grid, [(0, 0)] * (grid.ndim - 2) + [(1, 1), (1, 1)])
        total = (
            padded[..., 1:-1, 1:-1]
            + padded[..., 1:-1, :-2]
            + padded[..., 1:-1, 2:]
            + padded[..., :-2, 1:-1]
            + padded[..., 2:, 1:-1]
        )
        return total.reshape(per_site.shape)

    def wall_coordinate(self) -> np.ndarray:
        """The coordinate a wall travels along, per site: x, so a straight wall enters from one edge."""
        x, _y = self._xy()
        return x.astype(float)

    def wall_extent(self) -> float:
        """The length a wall crosses, in sites: the width."""
        return float(self.width)

    def energy_batched(self, spins: np.ndarray) -> np.ndarray:
        """The energy of each configuration in a stack ``(P, N, 3)``, J, shape ``(P,)``."""
        grid = self._grid(spins)
        anisotropy = -self.anisotropy_j * np.sum(spins[..., 2] ** 2, axis=-1)
        bonds = np.sum(grid[..., :, :-1, :] * grid[..., :, 1:, :], axis=(-1, -2, -3)) + np.sum(
            grid[..., :-1, :, :] * grid[..., 1:, :, :], axis=(-1, -2, -3)
        )
        return anisotropy - self.exchange_j * bonds

    def minus_energy_gradient_batched(self, spins: np.ndarray) -> np.ndarray:
        """``-dE/ds`` for a stack of configurations, J per unit vector, shape ``(P, N, 3)``."""
        return self.internal_field_batched(spins) * self.mu
