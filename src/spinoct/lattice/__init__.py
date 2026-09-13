"""Optimal control of magnetization switching beyond the macrospin: the spin chain.

The authors of the method state the open problem in print (Phys. Rev. B 107, 214448, 2023): the
macrospin approximation breaks down with size, and the transition may involve nonuniform rotation,
domain-wall nucleation and propagation, or spin waves, and "it remains to be seen under what conditions
these ... switching mechanisms become optimal in terms of energy efficiency."

This module takes the first step past the macrospin that has been taken anywhere except a single 1D
study: a ferromagnetic spin chain with nearest-neighbour exchange, where the reversal can be uniform
(every spin rotates together) or nonuniform (a domain wall sweeps through). It computes the switching
cost of each mode and finds the crossover, the size and exchange at which nonuniform switching becomes
cheaper.
"""

from __future__ import annotations

from .chain import SpinChain
from .reversal import ReversalComparison, compare_reversal_modes, domain_wall_cost, uniform_cost

__all__ = ["ReversalComparison", "SpinChain", "compare_reversal_modes", "domain_wall_cost", "uniform_cost"]
