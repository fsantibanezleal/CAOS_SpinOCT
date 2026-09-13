"""A ferromagnetic spin chain with nearest-neighbour exchange and uniaxial anisotropy.

The energy of a chain of ``N`` unit moments is

    E = -K sum_i s_{i,z}^2  -  J sum_{<ij>} s_i . s_j

with ``K`` the easy-axis anisotropy per site (joules) and ``J`` the nearest-neighbour exchange
(joules, positive is ferromagnetic). The internal field at site ``i`` is ``-(1/mu) dE/ds_i``, which
adds an exchange term ``(J/mu)(s_{i-1} + s_{i+1})`` to the single-site anisotropy field. Open boundary
conditions (the end sites have one neighbour).

This is the same physics as the macrospin, one per site, coupled by exchange. In the limit of strong
exchange relative to the anisotropy the chain locks into a single macrospin and the macrospin results
are recovered, which is the validation limit for the lattice solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T

__all__ = ["SpinChain"]


@dataclass(frozen=True)
class SpinChain:
    """A 1D ferromagnetic chain of unit moments.

    Attributes:
        n_sites: the number of sites ``N``.
        mu: the magnetic moment per site, J/T.
        anisotropy_j: the easy-axis anisotropy per site ``K``, J.
        exchange_j: the nearest-neighbour exchange ``J``, J. Positive is ferromagnetic.
        alpha: Gilbert damping, dimensionless.
        gamma: gyromagnetic ratio, rad/(s T).
        label: a short display name.
    """

    n_sites: int
    mu: float
    anisotropy_j: float
    exchange_j: float
    alpha: float
    gamma: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T
    label: str = "chain"

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be at least 1")
        if self.mu <= 0.0 or self.anisotropy_j <= 0.0:
            raise ValueError("mu and anisotropy_j must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")

    @property
    def tau0(self) -> float:
        """The single-site Larmor timescale ``mu / (2 gamma K)``, s."""
        return self.mu / (2.0 * self.gamma * self.anisotropy_j)

    @property
    def anisotropy_field(self) -> float:
        """The single-site anisotropy field ``K / mu``, T."""
        return self.anisotropy_j / self.mu

    @property
    def exchange_field(self) -> float:
        """The exchange field scale ``J / mu``, T."""
        return self.exchange_j / self.mu

    def energy(self, spins: np.ndarray) -> float:
        """Total energy of a configuration, J.

        Args:
            spins: unit moments, shape ``(N, 3)``.

        Returns:
            The energy in J.
        """
        spins = np.asarray(spins, dtype=float)
        anisotropy = -self.anisotropy_j * np.sum(spins[:, 2] ** 2)
        exchange = -self.exchange_j * np.sum(np.sum(spins[:-1] * spins[1:], axis=-1))
        return float(anisotropy + exchange)

    def internal_field(self, spins: np.ndarray) -> np.ndarray:
        """The internal field at every site, ``-(1/mu) dE/ds_i``, shape ``(N, 3)``, T.

        Args:
            spins: unit moments, shape ``(N, 3)``.

        Returns:
            The internal field per site, T.
        """
        spins = np.asarray(spins, dtype=float)
        out = np.zeros_like(spins)
        # Anisotropy (easy z axis).
        out[:, 2] += 2.0 * self.anisotropy_j / self.mu * spins[:, 2]
        # Exchange from neighbours, open boundaries.
        exchange_coeff = self.exchange_j / self.mu
        out[:-1] += exchange_coeff * spins[1:]
        out[1:] += exchange_coeff * spins[:-1]
        return out

    def internal_field_transverse(self, spins: np.ndarray) -> np.ndarray:
        """The per-site internal field with its component along each moment removed, shape ``(N, 3)``."""
        spins = np.asarray(spins, dtype=float)
        field = self.internal_field(spins)
        longitudinal = np.sum(field * spins, axis=-1, keepdims=True)
        return field - longitudinal * spins
