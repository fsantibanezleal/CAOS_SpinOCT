"""The Landau-Lifshitz-Gilbert equation, its inverse, and norm-preserving integration.

The equation of motion, in the Gilbert form used throughout the optimal-control literature for
magnetization switching (Phys. Rev. Lett. 126, 177206 (2021), equation 2):

    (1 + alpha^2) s' = -gamma s x (b_i + b) - alpha gamma s x [ s x (b_i + b) ]

with ``s`` the unit moment direction, ``b_i`` the internal field from the anisotropy and ``b`` the
applied control field, both in tesla.

The inverse relation is what makes the optimal control problem tractable. Given a trajectory, the
field that produces it is determined:

    b(s, s') = (alpha / gamma) s' + (1 / gamma) [ s x s' ] - b_i_perp

Substituting it into ``Phi = int |b|^2 dt`` converts a constrained optimization over ``(b, s)`` into
an unconstrained one over ``s`` alone. Only the transverse part of the internal field appears,
because the longitudinal part does not affect the dynamics.

Integration preserves ``|s| = 1`` exactly by renormalizing after each stage. A plain Cartesian
integrator drifts off the sphere and silently changes the answer over the thousands of Larmor
periods a switching protocol spans.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .system import MacrospinSystem

__all__ = ["field_from_trajectory", "integrate_llg", "llg_rhs", "switching_cost"]


def llg_rhs(s: np.ndarray, b_applied: np.ndarray, system: MacrospinSystem) -> np.ndarray:
    """The right-hand side of the Landau-Lifshitz-Gilbert equation, ``ds/dt``.

    Args:
        s: unit moment directions, shape ``(..., 3)``.
        b_applied: applied control field in T, shape ``(..., 3)``.
        system: the macrospin, supplying ``alpha``, ``gamma`` and the internal field.

    Returns:
        ``ds/dt`` in 1/s, shape ``(..., 3)``.
    """
    s = np.asarray(s, dtype=float)
    b_total = system.internal_field(s) + np.asarray(b_applied, dtype=float)
    alpha, gamma = system.alpha, system.gamma
    precession = np.cross(s, b_total)
    damping = np.cross(s, precession)
    return (-gamma * precession - alpha * gamma * damping) / (1.0 + alpha**2)


def field_from_trajectory(
    s: np.ndarray, s_dot: np.ndarray, system: MacrospinSystem
) -> np.ndarray:
    """Invert the equation of motion: the field that produces a given trajectory.

    ``b = (alpha / gamma) s' + (1 / gamma) [s x s'] - b_i_perp``.

    Args:
        s: unit moment directions, shape ``(..., 3)``.
        s_dot: time derivatives in 1/s, shape ``(..., 3)``.
        system: the macrospin.

    Returns:
        The applied field in T, shape ``(..., 3)``. It is perpendicular to ``s`` by construction
        whenever ``s_dot`` is, which it is for any trajectory that stays on the unit sphere.

    Notes:
        This is the single step that makes the optimal control problem unconstrained, and it is why
        the optimal field is always transverse. A longitudinal component is invisible here and can
        be added afterwards for dynamical stabilization, at a cost the functional does see.
    """
    s = np.asarray(s, dtype=float)
    s_dot = np.asarray(s_dot, dtype=float)
    alpha, gamma = system.alpha, system.gamma
    return (alpha / gamma) * s_dot + (1.0 / gamma) * np.cross(s, s_dot) - system.internal_field_transverse(s)


def switching_cost(times: np.ndarray, b_applied: np.ndarray) -> float:
    """The switching cost ``Phi = int |b|^2 dt`` by trapezoidal quadrature.

    Args:
        times: sample times in s, shape ``(N,)``, strictly increasing.
        b_applied: applied field in T, shape ``(N, 3)`` or ``(N,)`` for a bare amplitude.

    Returns:
        The cost in T^2 s. **Not** an energy; see :mod:`spinoct.units`.
    """
    times = np.asarray(times, dtype=float)
    b = np.asarray(b_applied, dtype=float)
    squared = b**2 if b.ndim == 1 else np.sum(b**2, axis=-1)
    return float(np.trapezoid(squared, times))


def integrate_llg(
    s0: np.ndarray,
    field: Callable[[float], np.ndarray],
    times: np.ndarray,
    system: MacrospinSystem,
) -> np.ndarray:
    """Integrate the equation of motion under a prescribed field, preserving the norm exactly.

    A fourth-order Runge-Kutta step followed by renormalization. Renormalizing is not a cosmetic
    correction: the constraint ``|s| = 1`` is exact in the physics, and letting it drift over the
    thousands of Larmor periods a protocol spans changes the reported switching outcome.

    Args:
        s0: the initial unit moment direction, shape ``(3,)``.
        field: a callable mapping time in s to the applied field in T, shape ``(3,)``.
        times: the output grid in s, shape ``(N,)``, strictly increasing and starting at the initial
            time. The grid is also the integration grid, so it must be fine enough to resolve the
            Larmor period; ``spinoct`` does not silently subdivide it.
        system: the macrospin.

    Returns:
        The trajectory, shape ``(N, 3)``, with ``out[0] == s0 / |s0|``.

    Raises:
        ValueError: if the grid is not strictly increasing.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional grid with at least two points")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")

    out = np.empty((times.size, 3), dtype=float)
    s = np.asarray(s0, dtype=float)
    s = s / np.linalg.norm(s)
    out[0] = s

    for index in range(times.size - 1):
        t0 = times[index]
        step = times[index + 1] - t0
        k1 = llg_rhs(s, field(t0), system)
        k2 = llg_rhs(s + 0.5 * step * k1, field(t0 + 0.5 * step), system)
        k3 = llg_rhs(s + 0.5 * step * k2, field(t0 + 0.5 * step), system)
        k4 = llg_rhs(s + step * k3, field(t0 + step), system)
        s = s + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s = s / np.linalg.norm(s)
        out[index + 1] = s

    return out
