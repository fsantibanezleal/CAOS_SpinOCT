"""Numerical optimal control path solvers.

The image-based direct-minimization method of Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B 107,
214448 (2023), reimplemented and validated against the closed-form uniaxial solution in
:mod:`spinoct.analytic`.
"""

from __future__ import annotations

from .image_ocp import ImageOCPResult, ImageOCPSolver

__all__ = ["ImageOCPResult", "ImageOCPSolver"]
