"""Closed-form optimal control paths. These are the package's positive controls."""

from __future__ import annotations

from .sot import SOTOptimalControl, ideal_sot_ratio_beta
from .uniaxial import (
    SwitchingTimeTooLongError,
    UniaxialOptimalControl,
    cost_free_macrospin,
    cost_infinite_time,
    optimal_switching_time,
    solve_shape_parameter,
)

__all__ = [
    "SOTOptimalControl",
    "SwitchingTimeTooLongError",
    "ideal_sot_ratio_beta",
    "UniaxialOptimalControl",
    "cost_free_macrospin",
    "cost_infinite_time",
    "optimal_switching_time",
    "solve_shape_parameter",
]
