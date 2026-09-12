"""The macrospin system: energy, internal field, Hessian, and the derived timescales.

Sign and axis convention
------------------------
The biaxial energy follows Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B 107, 214448 (2023),
https://doi.org/10.1103/PhysRevB.107.214448, equation (17):

    E(s) = xi * K * s_x^2 - K * s_z^2

so the **easy axis is z**, the **hard axis is x**, ``K > 0`` is the easy-axis anisotropy energy in
joules, and ``xi >= 0`` is the dimensionless hard-axis ratio. The energy minima sit at
``s = (0, 0, +1)`` and ``s = (0, 0, -1)``; the saddle points sit at ``s = (0, +1, 0)`` and
``s = (0, -1, 0)``.

The barrier between the minima is ``K`` **irrespective of xi**. That is the whole point of the
biaxial construction: the hard axis changes the switching cost without changing the thermal
stability, which dissolves the writability-versus-stability dilemma of magnetic memory.

Setting ``xi = 0`` recovers the uniaxial system ``E = -K s_z^2`` that the analytic solution of
Phys. Rev. Lett. 126, 177206 (2021) covers exactly.

Units are the package contract in :mod:`spinoct.units`: ``mu`` in J/T, ``gamma`` in rad/(s T),
``K`` in J, ``alpha`` and ``xi`` dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T

__all__ = ["MacrospinSystem"]


@dataclass(frozen=True)
class MacrospinSystem:
    """A single magnetic moment with biaxial anisotropy.

    Attributes:
        mu: magnetic moment magnitude, J/T.
        anisotropy_j: the easy-axis anisotropy energy ``K``, J. Must be positive.
        alpha: Gilbert damping, dimensionless. Must be non-negative.
        hard_axis_ratio: ``xi``, the hard-axis anisotropy relative to ``K``, dimensionless.
            Zero gives the uniaxial system. Must be non-negative.
        gamma: gyromagnetic ratio, rad/(s T). Defaults to the electron value.
        label: a short human-readable name carried into artifacts and figures.
    """

    mu: float
    anisotropy_j: float
    alpha: float
    hard_axis_ratio: float = 0.0
    gamma: float = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T
    label: str = "macrospin"

    def __post_init__(self) -> None:
        if self.mu <= 0.0:
            raise ValueError("mu must be positive (J/T)")
        if self.anisotropy_j <= 0.0:
            raise ValueError("anisotropy_j (K) must be positive (J)")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.hard_axis_ratio < 0.0:
            raise ValueError("hard_axis_ratio (xi) must be non-negative")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive (rad/(s T))")

    # ---------------------------------------------------------------- derived scales

    @property
    def tau0(self) -> float:
        """The Larmor timescale ``tau0 = mu / (2 gamma K)``, in seconds.

        Every switching time in this package is naturally expressed in units of ``tau0``; the
        published results are all quoted that way.
        """
        return self.mu / (2.0 * self.gamma * self.anisotropy_j)

    @property
    def anisotropy_field(self) -> float:
        """The anisotropy field ``K / mu``, in tesla.

        This is the natural field scale of the problem and the unit the published pulse amplitudes
        are quoted in.
        """
        return self.anisotropy_j / self.mu

    @property
    def energy_barrier(self) -> float:
        """The barrier between the two minima, in joules.

        Equals ``K`` for any ``xi``: the minimum is ``-K`` at the poles and the saddle is ``0`` at
        ``s = (0, +-1, 0)``.
        """
        return self.anisotropy_j

    def thermal_stability_factor(self, temperature_k: float) -> float:
        """The ratio of the barrier to the thermal energy, ``Delta = Delta_E / (k_B T)``.

        Args:
            temperature_k: temperature in K.

        Returns:
            The dimensionless stability factor. Magnetic memory conventionally requires at least 60
            for ten-year retention.
        """
        from ..units import thermal_energy_j

        return self.energy_barrier / thermal_energy_j(temperature_k)

    # ---------------------------------------------------------------- fields

    def energy(self, s: np.ndarray) -> np.ndarray:
        """Internal energy of a configuration, excluding the Zeeman term.

        Args:
            s: unit moment directions, shape ``(..., 3)``.

        Returns:
            Energy in J, shape ``(...)``.
        """
        s = np.asarray(s, dtype=float)
        return self.hard_axis_ratio * self.anisotropy_j * s[..., 0] ** 2 - self.anisotropy_j * s[..., 2] ** 2

    def internal_field(self, s: np.ndarray) -> np.ndarray:
        """The internal (anisotropy) field ``b_i = -(1/mu) dE/ds``.

        Args:
            s: unit moment directions, shape ``(..., 3)``.

        Returns:
            Field in T, shape ``(..., 3)``.
        """
        s = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        scale = 2.0 * self.anisotropy_j / self.mu
        out[..., 0] = -scale * self.hard_axis_ratio * s[..., 0]
        out[..., 2] = scale * s[..., 2]
        return out

    def internal_field_transverse(self, s: np.ndarray) -> np.ndarray:
        """The component of the internal field perpendicular to the moment.

        The longitudinal component does not affect the dynamics and does not enter the optimal
        control functional, so it is projected out here once rather than in every caller.

        Args:
            s: unit moment directions, shape ``(..., 3)``.

        Returns:
            Field in T, shape ``(..., 3)``.
        """
        s = np.asarray(s, dtype=float)
        b_i = self.internal_field(s)
        longitudinal = np.sum(b_i * s, axis=-1, keepdims=True)
        return b_i - longitudinal * s

    def hessian(self) -> np.ndarray:
        """The Hessian of the energy with respect to the Cartesian moment components.

        Returns:
            A ``(3, 3)`` array in J. Constant for a quadratic anisotropy, which is why it takes no
            argument; the Euler-Lagrange equation of the optimal control problem needs it.
        """
        return np.diag(
            [
                2.0 * self.hard_axis_ratio * self.anisotropy_j,
                0.0,
                -2.0 * self.anisotropy_j,
            ]
        )

    # ---------------------------------------------------------------- convenience

    def switching_time_from_tau0(self, multiples: float) -> float:
        """Convert a switching time expressed in ``tau0`` into seconds.

        Args:
            multiples: the switching time as a multiple of ``tau0``.

        Returns:
            The time in s.
        """
        return multiples * self.tau0

    def describe(self) -> dict[str, float | str]:
        """A flat, serializable summary of the system and its derived scales.

        Returns:
            A dictionary whose keys carry their unit in the name, so an artifact written from it is
            self-describing.
        """
        return {
            "label": self.label,
            "mu_j_per_t": self.mu,
            "anisotropy_j": self.anisotropy_j,
            "alpha": self.alpha,
            "hard_axis_ratio": self.hard_axis_ratio,
            "gamma_rad_per_s_t": self.gamma,
            "tau0_s": self.tau0,
            "anisotropy_field_t": self.anisotropy_field,
            "energy_barrier_j": self.energy_barrier,
        }
