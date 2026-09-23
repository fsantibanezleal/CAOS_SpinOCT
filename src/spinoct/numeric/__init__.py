"""Numerical optimal control path solvers.

The image-based direct-minimization method of Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B 107,
214448 (2023), reimplemented and validated against the closed-form uniaxial solution in
:mod:`spinoct.analytic`.

:class:`ImageOCPSolver` is the reference lane and solves one problem on the CPU.
:class:`BatchImageOCPSolver` solves a batch of them as one tensor, on a GPU when there is one; it needs
the optional ``spinoct[torch]`` extra, and it is accepted only against the CPU lane's answers.
"""

from __future__ import annotations

from .batch_ocp import BatchImageOCPSolver, BatchOCPResult, torch_is_available
from .image_ocp import ImageOCPResult, ImageOCPSolver

__all__ = [
    "BatchImageOCPSolver",
    "BatchOCPResult",
    "ImageOCPResult",
    "ImageOCPSolver",
    "torch_is_available",
]
