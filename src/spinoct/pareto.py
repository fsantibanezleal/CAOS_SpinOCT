"""Multi-objective trade-offs: the Pareto front of switching protocols (R14).

The literature reports single scalar optima. The real device question is a surface in several
objectives at once: the switching time, the cost, the peak field a generator must supply, and the
bandwidth it must span. A protocol is Pareto-optimal if no other protocol beats it on every objective
simultaneously. This module computes that front over the analytic optimal-control family, so the
trade-offs are explicit rather than hidden behind one scalar.

Objectives (all to be minimized):

- switching time ``T``;
- switching cost ``Phi``;
- peak field amplitude ``b_max``;
- pulse bandwidth (the 99 percent spectral energy width).

The knee time ``T*`` of the cost-versus-time trade is a single point on this front; the front is the
whole picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic.uniaxial import UniaxialOptimalControl
from .dynamics.system import MacrospinSystem
from .metrics import pulse_bandwidth_fraction

__all__ = ["ParetoPoint", "pareto_front", "sweep_objectives"]


@dataclass(frozen=True)
class ParetoPoint:
    """One protocol evaluated on every objective.

    Attributes:
        switching_time_tau0: the switching time in units of tau0.
        switching_time_s: the switching time in seconds.
        cost: the switching cost, T^2 s.
        peak_field: the peak field amplitude, T.
        bandwidth_hz: the 99 percent spectral bandwidth, Hz.
        dominated: whether some other point beats this one on every objective.
    """

    switching_time_tau0: float
    switching_time_s: float
    cost: float
    peak_field: float
    bandwidth_hz: float
    dominated: bool


def sweep_objectives(
    system: MacrospinSystem,
    switching_times_tau0: np.ndarray,
    samples: int = 2000,
) -> list[ParetoPoint]:
    """Evaluate the analytic optimal pulse on every objective across a switching-time sweep.

    Args:
        system: the macrospin (uniaxial).
        switching_times_tau0: the switching times to evaluate, in units of tau0.
        samples: the time resolution for the peak and bandwidth.

    Returns:
        A list of :class:`ParetoPoint`, dominance not yet marked (all ``dominated=False``).
    """
    points: list[ParetoPoint] = []
    for t_tau0 in switching_times_tau0:
        switching_time = system.switching_time_from_tau0(float(t_tau0))
        optimal = UniaxialOptimalControl.for_switching_time(system, switching_time)
        grid = np.linspace(0.0, switching_time, samples)
        amplitude = optimal.field_amplitude(grid)
        field = optimal.field_vector(grid)
        points.append(
            ParetoPoint(
                switching_time_tau0=float(t_tau0),
                switching_time_s=switching_time,
                cost=optimal.cost(),
                peak_field=float(np.max(np.abs(amplitude))),
                bandwidth_hz=pulse_bandwidth_fraction(grid, field),
                dominated=False,
            )
        )
    return points


def _objective_vector(point: ParetoPoint) -> np.ndarray:
    return np.array([point.switching_time_s, point.cost, point.peak_field, point.bandwidth_hz])


def pareto_front(
    system: MacrospinSystem,
    switching_times_tau0: np.ndarray | None = None,
    samples: int = 2000,
) -> list[ParetoPoint]:
    """The Pareto front over (switching time, cost, peak field, bandwidth) for the optimal family.

    Args:
        system: the macrospin (uniaxial).
        switching_times_tau0: the switching times to consider; a log-spaced default is used if omitted.
        samples: the time resolution.

    Returns:
        Every evaluated point, each marked ``dominated`` or not. The non-dominated points are the
        Pareto front. A point A dominates B if A is at least as good on every objective and strictly
        better on at least one.
    """
    if switching_times_tau0 is None:
        switching_times_tau0 = np.geomspace(0.5, 200.0, 24)

    points = sweep_objectives(system, np.asarray(switching_times_tau0), samples=samples)
    vectors = np.stack([_objective_vector(p) for p in points])

    marked: list[ParetoPoint] = []
    for i, point in enumerate(points):
        others = np.delete(vectors, i, axis=0)
        # Dominated if some other point is <= on all objectives and < on at least one.
        at_least_as_good = np.all(others <= vectors[i] + 1e-18, axis=1)
        strictly_better = np.any(others < vectors[i] - 1e-18, axis=1)
        dominated = bool(np.any(at_least_as_good & strictly_better))
        marked.append(
            ParetoPoint(
                switching_time_tau0=point.switching_time_tau0,
                switching_time_s=point.switching_time_s,
                cost=point.cost,
                peak_field=point.peak_field,
                bandwidth_hz=point.bandwidth_hz,
                dominated=dominated,
            )
        )
    return marked
