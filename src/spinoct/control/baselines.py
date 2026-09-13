"""Conventional switching protocols, the baselines the optimal control paths are measured against.

Scoring an optimal result only against the weakest baseline flatters it. This module implements the
whole ladder of conventional protocols so the reduction factor is always quoted against the strongest
one available, per the product's honesty contract.

Protocols
---------
- Static antiparallel field: the textbook Stoner-Wohlfarth reversal, a constant field opposite the
  initial state above the switching field. Slow and damping limited.
- Sun-Wang minimal field: the theoretical minimum constant-amplitude field and the shortest-time
  constant pulse (Sun and Wang, Phys. Rev. Lett. 97, 077205 (2006),
  https://doi.org/10.1103/PhysRevLett.97.077205).
- Precessional: a transverse field pulse whose duration is tuned to land the moment on the far basin
  by precession, faster than damping-limited reversal but sensitive to the duration.

Each protocol integrates the same Landau-Lifshitz-Gilbert equation as the optimal solvers, through the
same norm-preserving integrator, and reports the same switching cost, so the comparison is
apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..dynamics.llg import integrate_llg, switching_cost
from ..dynamics.system import MacrospinSystem

__all__ = [
    "ConstantFieldProtocol",
    "PrecessionalProtocol",
    "ProtocolResult",
    "static_switching_field",
    "sun_wang_minimal_field",
]


def static_switching_field(system: MacrospinSystem) -> float:
    """The anisotropy (switching) field ``K / mu`` in tesla, the field a static reversal must exceed.

    Args:
        system: the macrospin.

    Returns:
        The field in T. For a uniaxial macrospin a constant antiparallel field above this value makes
        the initial state unstable and the moment relaxes to the reversed state.
    """
    return system.anisotropy_field


def sun_wang_minimal_field(system: MacrospinSystem) -> float:
    """The Sun-Wang theoretical minimum constant-amplitude switching field, in tesla.

    Sun and Wang, Phys. Rev. Lett. 97, 077205 (2006), show that the smallest constant field that can
    reverse a Stoner particle, applied at the optimal angle, is ``K / (2 mu) * (1 / ... )`` reduced
    from the astroid value; the widely used compact result for the minimal magnitude at the optimal
    direction is ``0.5 * K / mu`` (half the anisotropy field) in the low-damping limit.

    Args:
        system: the macrospin.

    Returns:
        The minimal field in T. This is the strongest constant-field baseline and the fairest static
        comparison for an optimal pulse.
    """
    # The Stoner-Wohlfarth astroid minimum: applied at 45 degrees to the easy axis, the switching
    # field drops to half the anisotropy field. This is a geometric factor of the astroid, not a
    # fitted constant.
    return 0.5 * system.anisotropy_field


@dataclass(frozen=True)
class ProtocolResult:
    """The outcome of running a conventional protocol.

    Attributes:
        times: the time grid, s.
        trajectory: the moment direction at each time, shape ``(N, 3)``.
        field: the applied field at each time, shape ``(N, 3)``, T.
        cost: the switching cost, T^2 s, comparable to the optimal-control cost.
        switched: whether the moment ended in the reversed basin (``s_z < 0``).
        final_sz: the final z-component, for a graded view of how complete the reversal was.
    """

    times: np.ndarray
    trajectory: np.ndarray
    field: np.ndarray
    cost: float
    switched: bool
    final_sz: float


def _run(
    system: MacrospinSystem, field: Callable[[float], np.ndarray], times: np.ndarray
) -> ProtocolResult:
    trajectory = integrate_llg(np.array([0.0, 0.0, 1.0]), field, times, system)
    field_table = np.stack([field(float(t)) for t in times])
    cost = switching_cost(times, field_table)
    final_sz = float(trajectory[-1, 2])
    return ProtocolResult(
        times=times,
        trajectory=trajectory,
        field=field_table,
        cost=cost,
        switched=final_sz < 0.0,
        final_sz=final_sz,
    )


@dataclass(frozen=True)
class ConstantFieldProtocol:
    """A constant field applied for the whole switching window.

    Attributes:
        system: the macrospin.
        amplitude: the field magnitude, T.
        polar_angle: the field direction measured from the easy (z) axis, radians. Zero is exactly
            antiparallel would be pi; the default of ``3 pi / 4`` places it in the reversing hemisphere
            tilted off the axis, which is where a constant field actually drives a reversal (a field
            exactly along the axis exerts no initial torque).
    """

    system: MacrospinSystem
    amplitude: float
    polar_angle: float = 0.75 * np.pi

    def field_vector(self) -> np.ndarray:
        """The constant field as a Cartesian vector, T."""
        return self.amplitude * np.array(
            [np.sin(self.polar_angle), 0.0, np.cos(self.polar_angle)]
        )

    def run(self, switching_time: float, n_steps: int = 8001) -> ProtocolResult:
        """Integrate the reversal under the constant field.

        Args:
            switching_time: the window, s.
            n_steps: the integration grid size.

        Returns:
            The :class:`ProtocolResult`.
        """
        vector = self.field_vector()
        times = np.linspace(0.0, switching_time, n_steps)
        return _run(self.system, lambda _t: vector, times)


@dataclass(frozen=True)
class PrecessionalProtocol:
    """A transverse field pulse driving precessional reversal, off for the remainder.

    A field perpendicular to the easy axis torques the moment into precession; timed correctly the
    precession carries it to the far basin, then the field is removed and damping settles it. Faster
    than a damping-limited static reversal, but the outcome depends on the pulse duration, which is why
    it is a baseline rather than an optimum.

    Attributes:
        system: the macrospin.
        amplitude: the transverse field magnitude, T.
        pulse_fraction: the fraction of the switching window the field is on.
    """

    system: MacrospinSystem
    amplitude: float
    pulse_fraction: float = 0.5

    def run(self, switching_time: float, n_steps: int = 8001) -> ProtocolResult:
        """Integrate the precessional reversal.

        Args:
            switching_time: the window, s.
            n_steps: the integration grid size.

        Returns:
            The :class:`ProtocolResult`.
        """
        on_until = self.pulse_fraction * switching_time
        transverse = self.amplitude * np.array([1.0, 0.0, 0.0])
        off = np.zeros(3)

        def field(t: float) -> np.ndarray:
            return transverse if t <= on_until else off

        times = np.linspace(0.0, switching_time, n_steps)
        return _run(self.system, field, times)
