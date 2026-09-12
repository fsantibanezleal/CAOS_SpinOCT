"""System definitions and the Landau-Lifshitz-Gilbert equation of motion."""

from __future__ import annotations

from .llg import field_from_trajectory, llg_rhs
from .system import MacrospinSystem

__all__ = ["MacrospinSystem", "field_from_trajectory", "llg_rhs"]
