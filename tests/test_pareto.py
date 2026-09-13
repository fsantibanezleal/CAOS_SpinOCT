"""The multi-objective Pareto front (R14)."""

from __future__ import annotations

import numpy as np

from spinoct.dynamics import MacrospinSystem
from spinoct.pareto import pareto_front, sweep_objectives
from spinoct.units import bohr_magnetons_to_j_per_t, mev_to_joules


def make_system() -> MacrospinSystem:
    return MacrospinSystem(mu=bohr_magnetons_to_j_per_t(3.0), anisotropy_j=mev_to_joules(0.15), alpha=0.1)


def test_objectives_trade_off_monotonically_with_switching_time() -> None:
    """Longer switching time is cheaper, gentler in peak field, and narrower in bandwidth."""
    system = make_system()
    points = sweep_objectives(system, np.geomspace(0.5, 100.0, 12))
    times = np.array([p.switching_time_s for p in points])
    peak = np.array([p.peak_field for p in points])
    bandwidth = np.array([p.bandwidth_hz for p in points])
    order = np.argsort(times)
    # Peak field falls strictly as the switching time grows; bandwidth falls as a trend (its FFT
    # estimate is discretized, so it is not strictly monotone step to step).
    assert np.all(np.diff(peak[order]) < 0)
    assert bandwidth[order][-1] < 0.2 * bandwidth[order][0]
    assert np.mean(np.diff(bandwidth[order]) < 0) >= 0.8


def test_the_whole_optimal_family_is_pareto_optimal_in_the_monotone_regime() -> None:
    """When switching time trades against every other objective, no point dominates another."""
    system = make_system()
    front = pareto_front(system, np.geomspace(1.0, 100.0, 14))
    non_dominated = [p for p in front if not p.dominated]
    # Every point offers a distinct trade of speed against cost/peak/bandwidth, so all survive.
    assert len(non_dominated) == len(front)


def test_faster_switching_costs_more_peak_field() -> None:
    system = make_system()
    front = pareto_front(system, np.array([2.0, 50.0]))
    fast = next(p for p in front if p.switching_time_tau0 == 2.0)
    slow = next(p for p in front if p.switching_time_tau0 == 50.0)
    assert fast.peak_field > slow.peak_field
    assert fast.bandwidth_hz > slow.bandwidth_hz
