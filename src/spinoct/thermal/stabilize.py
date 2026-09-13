"""Longitudinal stabilization of an optimal pulse, and the cost of reliability.

Source
------
Badarneh, Kwiatkowski and Bessarab, *Enhancing thermal stability of optimal magnetization reversal in
nanoparticles*, arXiv:2312.11293 (2023).

The problem
-----------
The optimal pulse is always perpendicular to the moment, so it does nothing to damp the perturbations
that thermal noise excites. Linearizing the dynamics about the optimal control path gives a
two-component perturbation whose growth is governed by the eigenvalues of the energy Hessian shifted by
the longitudinal field component ``B_r`` (the part of the applied field parallel to the moment):

    w1 = B_r + (K / mu) cos(2 theta)
    w2 = B_r + (K / mu) cos^2(theta)

Perturbations are bounded (elliptic) when ``w1 w2 > 0`` and divergent (hyperbolic) when ``w1 w2 <= 0``.
At ``B_r = 0`` a large part of the reversal, ``pi/4 <= theta <= 3 pi/4``, is hyperbolic, and that
hyperbolicity, not the energy barrier, is the primary cause of the pulse and the moment losing phase
lock. Adding a longitudinal field with ``B_r > K/mu`` or ``B_r < -K/mu`` removes the hyperbolic domain
and drives the switching success rate to unity.

But ``B_r`` is not free. The optimal pulse is perpendicular, so ``B_r`` is invisible to the
leading-order dynamics, yet it still costs ``integral B_r^2 dt`` in the switching cost. The trade
between reliability and cost is the object this module quantifies, and it is not in the literature.

Two things are provided:

- :func:`br_cost_reliability_front`, the Pareto front of longitudinal field: for a sweep of ``B_r``,
  the added cost and the Monte-Carlo success rate at temperature.
- :func:`instability_penalty`, the deterministic hyperbolicity integral along a path, which is the
  instability-penalized optimal control path's extra objective term; the claim to test is that it
  predicts the Monte-Carlo success rate without running an ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analytic.uniaxial import UniaxialOptimalControl
from ..dynamics.system import MacrospinSystem
from .stochastic import switching_success_rate

__all__ = [
    "BrFrontPoint",
    "br_cost_reliability_front",
    "hyperbolic_fraction",
    "instability_penalty",
    "perturbation_eigenvalues",
]


def perturbation_eigenvalues(
    theta: np.ndarray, longitudinal_field: float, system: MacrospinSystem
) -> tuple[np.ndarray, np.ndarray]:
    """The two perturbation eigenvalues along the path, in tesla.

    Args:
        theta: polar angle(s) along the trajectory, radians.
        longitudinal_field: ``B_r``, the applied-field component parallel to the moment, T.
        system: the macrospin.

    Returns:
        ``(w1, w2)``, each the same shape as ``theta``, in T.
    """
    theta = np.asarray(theta, dtype=float)
    anisotropy_field = system.anisotropy_field
    w1 = longitudinal_field + anisotropy_field * np.cos(2.0 * theta)
    w2 = longitudinal_field + anisotropy_field * np.cos(theta) ** 2
    return w1, w2


def hyperbolic_fraction(
    theta: np.ndarray, longitudinal_field: float, system: MacrospinSystem
) -> float:
    """The fraction of the path that is dynamically unstable (hyperbolic).

    Args:
        theta: polar angles along the trajectory, radians.
        longitudinal_field: ``B_r`` in T.
        system: the macrospin.

    Returns:
        The fraction of samples with ``w1 w2 <= 0``, between 0 and 1. Zero means the whole path is
        dynamically stable.
    """
    w1, w2 = perturbation_eigenvalues(theta, longitudinal_field, system)
    return float(np.mean(w1 * w2 <= 0.0))


def instability_penalty(
    theta: np.ndarray, times: np.ndarray, system: MacrospinSystem, longitudinal_field: float = 0.0
) -> float:
    """The hyperbolicity integral ``integral max(0, -w1 w2) dt`` along a path, in T^2 s.

    This is the extra objective term of the instability-penalized optimal control path. It costs
    nothing extra to evaluate, because ``w1`` and ``w2`` come from the same Hessian the solver already
    uses, and the claim to test is that sweeping its weight predicts the Monte-Carlo success rate.

    Args:
        theta: polar angles along the trajectory, radians.
        times: the sample times, s.
        system: the macrospin.
        longitudinal_field: ``B_r`` in T, usually zero for the bare optimal pulse.

    Returns:
        The penalty in T^2 s.
    """
    w1, w2 = perturbation_eigenvalues(theta, longitudinal_field, system)
    integrand = np.maximum(0.0, -(w1 * w2))
    return float(np.trapezoid(integrand, times))


@dataclass(frozen=True)
class BrFrontPoint:
    """One point on the longitudinal-field cost-reliability front.

    Attributes:
        longitudinal_field: ``B_r`` in T.
        longitudinal_field_over_anisotropy: ``B_r / (K/mu)``, dimensionless.
        added_cost: the extra switching cost from the longitudinal field, ``integral B_r^2 dt``, T^2 s.
        success_rate: the Monte-Carlo switching success rate at temperature.
        confidence95: the 95 percent binomial confidence half-width on the rate.
        hyperbolic_fraction: the fraction of the path that is dynamically unstable at this ``B_r``.
    """

    longitudinal_field: float
    longitudinal_field_over_anisotropy: float
    added_cost: float
    success_rate: float
    confidence95: float
    hyperbolic_fraction: float


def br_cost_reliability_front(
    system: MacrospinSystem,
    switching_time: float,
    temperature_k: float,
    br_over_anisotropy: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
    n_copies: int = 400,
    n_steps: int = 800,
    seed: int = 0,
) -> list[BrFrontPoint]:
    """Compute the cost-reliability front of the longitudinal field.

    For each ``B_r`` the bare optimal (uniaxial) pulse is applied together with a constant longitudinal
    field ``B_r`` along the instantaneous moment direction, an ensemble is run at temperature, and the
    added cost and success rate are recorded.

    Args:
        system: the macrospin. Uniaxial (the bare optimal pulse is the closed-form one).
        switching_time: ``T`` in s.
        temperature_k: temperature, K.
        br_over_anisotropy: the sweep of ``B_r`` in units of the anisotropy field ``K/mu``.
        n_copies: ensemble size per point.
        n_steps: integration steps.
        seed: base random seed; each point uses a distinct derived seed.

    Returns:
        A list of :class:`BrFrontPoint`, one per swept value.
    """
    optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)
    anisotropy_field = system.anisotropy_field
    grid = np.linspace(0.0, switching_time, n_steps + 1)
    theta_path = optimal.polar_angle(grid)

    front: list[BrFrontPoint] = []
    for offset, ratio in enumerate(br_over_anisotropy):
        br = ratio * anisotropy_field

        def field(t: float, br: float = br) -> np.ndarray:
            # The bare perpendicular optimal pulse plus a longitudinal component along the current
            # optimal-path moment direction (the stabilizing term of arXiv:2312.11293).
            perpendicular = optimal.field_vector(float(t))
            moment = optimal.moment(float(t))
            return perpendicular + br * moment

        result = switching_success_rate(
            system,
            field,
            switching_time,
            temperature_k,
            n_copies=n_copies,
            n_steps=n_steps,
            seed=seed + offset,
        )
        added_cost = float(br**2 * switching_time)
        front.append(
            BrFrontPoint(
                longitudinal_field=br,
                longitudinal_field_over_anisotropy=ratio,
                added_cost=added_cost,
                success_rate=result.success_rate,
                confidence95=result.confidence95,
                hyperbolic_fraction=hyperbolic_fraction(theta_path, br, system),
            )
        )
    return front
