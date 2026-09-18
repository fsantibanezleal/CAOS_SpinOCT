"""Exact-gradient optimization of a control that is LINEAR in its parameters.

Both constrained solvers parameterize the transverse field linearly: GRAPE interpolates piecewise
values from slice nodes onto the integration grid, and CRAB sums a few harmonics with fixed
frequencies. In both cases the field on the grid is ``design @ parameters`` for a fixed matrix, so the
exact gradient with respect to the parameters follows from the exact gradient with respect to the field
by one transposed multiply:

    dC/dp = design^T (dC/db)

and ``dC/db`` is what the discrete adjoint (:mod:`spinoct.adjoint`) returns in a single backward pass,
at the cost of one extra integration and independent of the number of parameters.

Why this module exists. The first versions of these solvers had no gradient: GRAPE ran L-BFGS-B on a
finite-difference gradient, which costs one forward integration per parameter and is noisy at the
integration tolerance, and CRAB ran a Nelder-Mead simplex, which does not converge at a few dozen
parameters. Measured on the uniaxial oracle at ten tau0, CRAB returned 2.2 times the analytic optimum at
two harmonics and 14 times at six, where a strictly larger search space cannot cost more. That
monotonicity is the falsifier, and it is what the gradient fixes.

The optimizer runs against the adjoint's own forward integration (norm-projected forward Euler), so the
objective and the gradient are a consistent pair, which is what a quasi-Newton method needs. The final
pulse is re-evaluated with the package's norm-preserving RK4 for reporting. The two agree: at the grid
used here the final ``s_z`` of the two integrators differs by about 4e-5, far below any reversal
threshold, and the reported cost ``Phi = int |b|^2 dt`` does not depend on the integrator at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..adjoint import adjoint_gradient
from ..dynamics.system import MacrospinSystem

__all__ = ["LinearControlProblem", "LinearControlSolution"]

#: L-BFGS-B convergence controls in the non-dimensionalized variables. The objective is scaled by the
#: free-macrospin cost and the field by the anisotropy field, so both are order one and these are real
#: tolerances rather than an accident of the physical units.
_FTOL = 1e-14
_GTOL = 1e-12
#: A reported cost must belong to a pulse that actually reversed the moment. The penalized objective
#: trades fidelity against cost, so a single solve at a fixed weight returns a cheap pulse that leaves
#: the moment part way: at one harmonic the first version of this stopped at an infidelity of 0.15,
#: which is s_z(T) = -0.69, and quoted its cost against an analytic optimum that reverses exactly.
#: 1e-5 rather than something looser: at 1e-3 the pulse stops 2.6e-4 short of the south pole and
#: banks the saving, reporting a cost one per cent BELOW the analytic optimum, which is the exact
#: number the optimum forbids for a complete reversal.
_TARGET_INFIDELITY = 1e-5
#: The weight is multiplied by this and the solve restarted from where it stopped, until the target is
#: met. Ten is aggressive enough to converge in a few rounds and gentle enough not to wreck the warm
#: start.
_PENALTY_GROWTH = 10.0
_MAX_PENALTY_ROUNDS = 8


@dataclass(frozen=True)
class LinearControlSolution:
    """The optimizer's answer, before a solver dresses it in its own result type.

    Attributes:
        parameters: the optimized parameter vector.
        field: the field table it produces, T.
        objective: the penalized objective at the solution, in physical units.
        iterations: the total L-BFGS-B iterations across the penalty rounds.
        converged: whether the optimizer converged AND the pulse met the reversal target.
        infidelity: the reversal infidelity reached, ``(1 + s_z(T)) / 2``.
        fidelity_weight: the penalty weight the answer was obtained at.
    """

    parameters: np.ndarray
    field: np.ndarray
    objective: float
    iterations: int
    converged: bool
    infidelity: float = 0.0
    fidelity_weight: float = 0.0


class LinearControlProblem:
    """A transverse control field that is a fixed linear map of its parameters.

    Args:
        system: the macrospin.
        times: the integration grid, s, shape ``(M,)``.
        design: the map from parameters to field samples, shape ``(M, P)``. The same matrix acts on the
            x and y parameter blocks.
        fidelity_weight: the penalty weight on an incomplete reversal, ``(1 + s_z(T)) / 2``.
        field_scale: the field unit the parameters are expressed in, T. Non-dimensionalizing matters:
            an objective of order 1e-12 meets L-BFGS's default tolerances before it takes a step.
        objective_scale: the cost unit the objective is expressed in, T^2 s.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        times: np.ndarray,
        design: np.ndarray,
        fidelity_weight: float,
        field_scale: float,
        objective_scale: float,
    ) -> None:
        self.system = system
        self.times = np.asarray(times, dtype=float)
        self.design = np.asarray(design, dtype=float)
        self.fidelity_weight = fidelity_weight
        self.field_scale = field_scale
        self.objective_scale = max(objective_scale, 1e-300)
        self.n_parameters = self.design.shape[1]

    def field_of(self, parameters: np.ndarray) -> np.ndarray:
        """The (M, 3) field table of a parameter vector, in tesla.

        The parameter vector holds the x block then the y block, each of ``n_parameters`` entries, in
        units of ``field_scale``.
        """
        blocks = parameters.reshape(2, self.n_parameters)
        transverse = self.design @ blocks.T * self.field_scale
        return np.concatenate([transverse, np.zeros((self.times.size, 1))], axis=1)

    def objective_and_gradient(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
        """The scaled objective and its exact gradient with respect to the parameters."""
        field = self.field_of(parameters)
        objective, _final_sz, grad_field = adjoint_gradient(
            field, self.times, self.system, self.fidelity_weight
        )
        # dC/dp = design^T dC/db, one block per transverse component, then back into parameter units.
        grad_blocks = self.design.T @ grad_field[:, :2]
        gradient = grad_blocks.T.reshape(-1) * self.field_scale / self.objective_scale
        return objective / self.objective_scale, gradient

    def infidelity_of(self, parameters: np.ndarray) -> float:
        """The reversal infidelity of a parameter vector, on the integration the gradient differentiates."""
        _objective, final_sz, _grad = adjoint_gradient(
            self.field_of(parameters), self.times, self.system, self.fidelity_weight
        )
        return 0.5 * (1.0 + final_sz)

    def solve(
        self,
        initial: np.ndarray,
        max_iterations: int,
        bounds: list[tuple[float, float]] | None = None,
        target_infidelity: float = _TARGET_INFIDELITY,
    ) -> LinearControlSolution:
        """Minimize with L-BFGS-B on the exact gradient, raising the penalty until the moment reverses.

        The objective trades the cost against the reversal penalty, so at a fixed weight the cheapest
        answer can be a pulse that leaves the moment part way. Each round solves to convergence, checks
        the infidelity, and if it is above the target multiplies the weight and restarts from the
        current point. A solution that never reaches the target is returned with ``converged`` false and
        its infidelity recorded, never silently.
        """
        vector = np.asarray(initial, dtype=float)
        weight = self.fidelity_weight
        iterations = 0
        success = False
        result = None
        for _round in range(_MAX_PENALTY_ROUNDS):
            self.fidelity_weight = weight
            result = minimize(
                self.objective_and_gradient,
                vector,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max_iterations, "ftol": _FTOL, "gtol": _GTOL},
            )
            vector = result.x
            iterations += int(result.nit)
            infidelity = self.infidelity_of(vector)
            if infidelity <= target_infidelity:
                success = bool(result.success)
                break
            weight *= _PENALTY_GROWTH
        else:
            infidelity = self.infidelity_of(vector)

        solution = LinearControlSolution(
            parameters=vector,
            field=self.field_of(vector),
            objective=float(result.fun) * self.objective_scale,
            iterations=iterations,
            converged=success and infidelity <= target_infidelity,
            infidelity=infidelity,
            fidelity_weight=weight,
        )
        self.fidelity_weight = weight
        return solution


def interpolation_design(nodes: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The matrix that linearly interpolates node values onto a grid, shape ``(len(grid), len(nodes))``.

    Built by interpolating the unit vectors, which is the definition of the linear map and cannot drift
    from whatever :func:`numpy.interp` does.
    """
    design = np.empty((grid.size, nodes.size))
    unit = np.zeros(nodes.size)
    for j in range(nodes.size):
        unit[:] = 0.0
        unit[j] = 1.0
        design[:, j] = np.interp(grid, nodes, unit)
    return design


def harmonic_design(frequencies: np.ndarray, grid: np.ndarray, switching_time: float) -> np.ndarray:
    """The CRAB basis on a grid: a sine AND a cosine per harmonic, windowed to vanish at both ends.

    Both quadratures are needed. With sine terms only, the x and y components of the field share a
    phase at every frequency, so the field direction is fixed in the plane and the pulse can only pulse,
    never rotate. The optimal uniaxial pulse is a rotating field, so a sine-only basis pays a large and
    entirely artificial penalty: at eight harmonics it sat at 6.2 times the analytic optimum, and adding
    harmonics did not help because no number of them can turn a linear drive into a rotating one.

    The window keeps the pulse from starting or ending abruptly, which a real generator would round off
    anyway, and is part of the basis rather than a post-processing step so the gradient sees it.

    Returns:
        The design matrix, shape ``(len(grid), 2 * len(frequencies))``: the sine columns first, then
        the cosine columns, so the bandwidth ceiling is still the highest frequency used.
    """
    envelope = np.sin(np.pi * grid / switching_time)
    phases = [2.0 * np.pi * f * grid for f in frequencies]
    columns = [np.sin(phase) * envelope for phase in phases]
    columns += [np.cos(phase) * envelope for phase in phases]
    return np.stack(columns, axis=1)
