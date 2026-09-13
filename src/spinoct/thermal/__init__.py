"""Thermal (stochastic) magnetization dynamics and reliability at temperature.

The optimal control paths are computed at zero temperature. Whether a pulse actually switches a bit at
a finite temperature is a separate, statistical question, and it is where the reliability of a protocol
is decided. This module adds the fluctuating thermal field to the Landau-Lifshitz-Gilbert equation
(the fluctuation-dissipation theorem fixes its strength), integrates an ensemble of trajectories, and
measures the switching success rate. It is the machinery behind the risk-aware objective (R11) and the
longitudinal-stabilization front (R12).
"""

from __future__ import annotations

from .stabilize import (
    BrFrontPoint,
    br_cost_reliability_front,
    hyperbolic_fraction,
    instability_penalty,
    perturbation_eigenvalues,
)
from .stochastic import (
    EnsembleResult,
    boltzmann_polar_variance,
    stochastic_llg_step,
    switching_success_rate,
)

__all__ = [
    "BrFrontPoint",
    "EnsembleResult",
    "boltzmann_polar_variance",
    "br_cost_reliability_front",
    "hyperbolic_fraction",
    "instability_penalty",
    "perturbation_eigenvalues",
    "stochastic_llg_step",
    "switching_success_rate",
]
