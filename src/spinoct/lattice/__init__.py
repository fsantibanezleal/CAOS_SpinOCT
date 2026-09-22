"""Optimal control of magnetization switching beyond the macrospin: the spin chain.

The authors of the method state the open problem in print (Phys. Rev. B 107, 214448, 2023): the
macrospin approximation breaks down with size, and the transition may involve nonuniform rotation,
domain-wall nucleation and propagation, or spin waves, and "it remains to be seen under what conditions
these ... switching mechanisms become optimal in terms of energy efficiency."

This package answers it for a ferromagnetic spin chain with nearest-neighbour exchange, in three
layers:

- :mod:`.reversal` compares two fixed reversal modes, uniform rotation and a constant-speed wall.
- :mod:`.ocp` minimizes the switching cost over every site's trajectory (the free optimal control
  path), bounded above by uniform rotation.
- :mod:`.mep` computes the minimum energy path, whose barrier sets a rigorous floor
  ``4 alpha Delta E / (gamma mu)`` under every pulse at every switching time.
"""

from __future__ import annotations

from .chain import SpinChain
from .mep import (
    MinimumEnergyPath,
    cost_floor_from_barrier,
    minimum_energy_path,
    recommended_images,
)
from .ocp import LatticeOCPResult, LatticeOCPSolver
from .patch import SpinPatch
from .reversal import ReversalComparison, compare_reversal_modes, domain_wall_cost, uniform_cost

__all__ = [
    "LatticeOCPResult",
    "LatticeOCPSolver",
    "MinimumEnergyPath",
    "ReversalComparison",
    "SpinChain",
    "SpinPatch",
    "compare_reversal_modes",
    "cost_floor_from_barrier",
    "domain_wall_cost",
    "minimum_energy_path",
    "recommended_images",
    "uniform_cost",
]
