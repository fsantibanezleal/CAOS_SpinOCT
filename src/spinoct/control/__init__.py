"""Conventional switching protocols and constrained optimal-control solvers.

The baselines every optimal result is measured against (static field, Sun-Wang optimal constant
field, precessional, microwave-assisted, DC spin-orbit torque), and the constrained pulse-shaping
solvers (GRAPE with amplitude and slew caps, CRAB in a band-limited basis).
"""

from __future__ import annotations

from .baselines import (
    ConstantFieldProtocol,
    PrecessionalProtocol,
    ProtocolResult,
    static_switching_field,
    sun_wang_minimal_field,
)

__all__ = [
    "ConstantFieldProtocol",
    "PrecessionalProtocol",
    "ProtocolResult",
    "static_switching_field",
    "sun_wang_minimal_field",
]
