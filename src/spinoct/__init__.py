"""spinoct: optimal control paths and energy-efficient switching pulses for classical spin dynamics.

The package solves one problem: given a magnetic system, an initial and a final state, and a
switching time, find the control (an applied magnetic field, an electric current, or both) that
drives the transition for the least dissipated energy.

It is the engine behind Espira (https://github.com/fsantibanezleal/CAOS_RES_Espira) and is
deliberately independent of any material database, so it can be used on any spin Hamiltonian.

Start here
----------
- :mod:`spinoct.units` is the dimensional contract. Read it before anything else; the same symbol
  means four different things across this literature.
- :mod:`spinoct.dynamics` carries the system definition and the Landau-Lifshitz-Gilbert equation.
- :mod:`spinoct.analytic` carries the closed-form optimal control paths, which are the positive
  controls every numerical result is checked against.
"""

from __future__ import annotations

__version__ = "0.3.0"
__display_version__ = "0.03.000"

__all__ = ["__display_version__", "__version__"]
