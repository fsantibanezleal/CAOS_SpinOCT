"""The exact optimal control protocol for spin-orbit-torque switching of a uniaxial nanomagnet.

Source
------
Vlasov, S. M., Kwiatkowski, G. J., Lobanov, I. S., Uzdin, V. M., Bessarab, P. F.,
*Optimal protocol for spin-orbit torque switching of a perpendicular nanomagnet*, Phys. Rev. B 105,
134404 (2022), https://doi.org/10.1103/PhysRevB.105.134404. Preprint arXiv:2203.01167.

The problem is the field analogue with the control being an in-plane electric current that acts on the
moment through field-like (FL) and damping-like (DL) spin-orbit torques. The cost is the Joule heating
``Phi = int |j|^2 dt``. The FL and DL coupling strengths are parameterized by their magnitude ``xi``
and a balance angle ``beta``:

    xi_F = xi cos(beta),   xi_D = xi sin(beta).

Closed-form results this module provides, each a positive control checked in
``tests/test_sot_analytic.py``:

- The minimum-cost asymptotics for fast and slow switching (Eqs. 9, 10).
- The average optimal current (Eq. 8), independent of the barrier height.
- The extremal current spread (Eq. 7).
- The characteristic switching time at the ideal SOT ratio (Eq. 12).
- The ideal ratio ``xi_D = -alpha xi_F`` (Eq. 11), for which the torque is entirely in the switching
  direction and the problem collapses onto the field-driven one, and the forbidden ratio
  ``xi_F = alpha xi_D`` for which the cost diverges (no switching).

Here ``eta = arctan(alpha)`` and ``K`` is the complete elliptic integral of the first kind (parameter
convention), and ``j0 = K_anis / (mu xi)`` is the natural current scale with ``K_anis`` the anisotropy
energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..dynamics.system import MacrospinSystem
from .elliptic import complete_k

__all__ = ["SOTOptimalControl", "ideal_sot_ratio_beta"]


def ideal_sot_ratio_beta(alpha: float) -> float:
    """The balance angle ``beta*`` giving the ideal SOT ratio ``xi_D = -alpha xi_F``.

    Args:
        alpha: Gilbert damping.

    Returns:
        ``beta*`` in radians, defined by ``tan(beta* + eta) = 0`` with ``eta = arctan(alpha)``, so
        ``beta* = -arctan(alpha)``. At this ratio the current torque points entirely along the
        switching direction and the optimal protocol is the simplest and cheapest.
    """
    return -math.atan(alpha)


@dataclass(frozen=True)
class SOTOptimalControl:
    """The exact optimal SOT switching protocol for a uniaxial macrospin.

    Attributes:
        system: the macrospin. Uniaxial; the closed form is for ``E = -K s_z^2``.
        switching_time: ``T`` in s.
        xi: the total SOT coupling magnitude, dimensionless in the reduced units of the reference.
        beta: the FL/DL balance angle in radians.
    """

    system: MacrospinSystem
    switching_time: float
    xi: float
    beta: float

    def __post_init__(self) -> None:
        if self.system.hard_axis_ratio != 0.0:
            raise ValueError("the closed-form SOT protocol covers the uniaxial case only")
        if self.xi <= 0.0:
            raise ValueError("the SOT coupling magnitude xi must be positive")

    @property
    def _eta(self) -> float:
        return math.atan(self.system.alpha)

    @property
    def current_scale(self) -> float:
        """The natural current scale ``j0 = K / (mu xi)`` in the reference's reduced units."""
        return self.system.anisotropy_j / (self.system.mu * self.xi)

    def is_forbidden(self) -> bool:
        """Whether the FL/DL ratio is the one that prohibits switching, ``xi_F = alpha xi_D``.

        Returns:
            True when ``tan(beta) = alpha`` (equivalently the torque has no component in the switching
            direction), for which the optimal cost diverges.
        """
        return math.isclose(math.tan(self.beta), self.system.alpha, rel_tol=1e-9, abs_tol=1e-12)

    def mean_current(self) -> float:
        """The time-averaged optimal current, Eq. 8, independent of the barrier height.

        ``<j> = 4 j0 sqrt(1 + alpha^2) K(sin^2(beta + eta)) / T``.

        Returns:
            The mean current in the reduced units.
        """
        alpha = self.system.alpha
        modulus_parameter = math.sin(self.beta + self._eta) ** 2
        return (
            4.0
            * self.current_scale
            * math.sqrt(1.0 + alpha**2)
            * float(complete_k(modulus_parameter))
            / self.switching_time
        )

    def cost_fast(self) -> float:
        """The fast-switching cost asymptote, Eq. 9, valid for ``T << (alpha + 1/alpha) tau0``.

        ``Phi ~ 4 (1 + alpha^2) K^2(sin^2(beta + eta)) / (T gamma^2 xi^2)``.

        Returns:
            The cost in the reduced units (current-squared integrated over time).
        """
        alpha = self.system.alpha
        modulus_parameter = math.sin(self.beta + self._eta) ** 2
        return (
            4.0
            * (1.0 + alpha**2)
            * float(complete_k(modulus_parameter)) ** 2
            / (self.switching_time * self.system.gamma**2 * self.xi**2)
        )

    def characteristic_time_ideal(self) -> float:
        """The characteristic switching time at the ideal ratio, Eq. 12, ``T0``.

        ``T0 = (1 + alpha^2) pi^2 tau0 / (2 alpha)``.

        Returns:
            The time in s. Below it the average current scales linearly in alpha, far below the
            conventional SOT critical current.
        """
        alpha = self.system.alpha
        if alpha <= 0.0:
            raise ValueError("the ideal characteristic time is undefined at zero damping")
        return (1.0 + alpha**2) * math.pi**2 * self.system.tau0 / (2.0 * alpha)

    def mean_current_ideal(self) -> float:
        """The average current at the ideal ratio, ``<j*> = 4 alpha j0 / (pi sqrt(1 + alpha^2))``.

        Returns:
            The mean current in the reduced units. Its linearity in alpha is the reason optimal SOT
            can undercut the conventional critical current at low damping.
        """
        alpha = self.system.alpha
        return 4.0 * alpha * self.current_scale / (math.pi * math.sqrt(1.0 + alpha**2))
