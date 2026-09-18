"""Constrained pulse-shaping optimal control: GRAPE and CRAB.

The analytic and image-based optimal control paths minimize the cost with no constraint on the
control, so the pulses they produce are mathematically optimal but not necessarily realizable: their
amplitude and phase vary on the Larmor timescale. Real pulse generators have finite bandwidth and a
finite peak amplitude. These solvers optimize the control **directly**, which is what lets a
constraint be imposed at all, and answer the question an experimentalist actually asks: what does the
cost become once the pulse must be realizable.

Two methods, both discretize-then-optimize:

- GRAPE (gradient ascent pulse engineering): the field is piecewise constant on N slices, and the
  slice values are optimized under a box constraint on the amplitude and an optional penalty on the
  slew rate. Standard in magnetic resonance, essentially unused for classical magnetization dynamics.
- CRAB (chopped random basis): the field is a truncated sum of a few randomized Fourier components, so
  the truncation itself is the bandwidth limit and the result is realizable by construction.

Both integrate the same Landau-Lifshitz-Gilbert equation as the rest of the package and report the
same switching cost, so their pulses sit on the same axes as the unconstrained optima. The objective
is the cost plus a penalty for an incomplete reversal:

    L = Phi + weight * infidelity,   infidelity = (1 + s_z(T)) / 2

which is zero for a complete reversal to the south pole and one for no reversal at all.

Both controls are LINEAR in their parameters, so both are optimized on the exact gradient from the
discrete adjoint (:mod:`spinoct.linear_basis`), in one backward pass whatever the parameter count. They
did not used to be: GRAPE ran L-BFGS-B on a finite-difference gradient and CRAB ran a Nelder-Mead
simplex, and at a few dozen parameters neither converged. The falsifier was monotonicity: measured on
the uniaxial oracle at ten tau0, CRAB returned 2.2 times the analytic optimum at two harmonics and 14
times at six, where a strictly larger search space cannot cost more. With the exact gradient the cost
falls as the bandwidth grows, which is the physics the sweep is meant to show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.llg import integrate_llg_tabulated, switching_cost
from ..dynamics.system import MacrospinSystem
from .linear_basis import LinearControlProblem, harmonic_design, interpolation_design

__all__ = ["ConstrainedResult", "CRABSolver", "GRAPESolver"]

#: A pulse counts as having reversed the moment when it ends within this of the south pole,
#: ``(1 + s_z(T)) / 2 <= 1e-3``, that is ``s_z(T) <= -0.998``. The sign of ``s_z`` alone is not
#: enough: a pulse that stops at ``s_z = -0.3`` is in the reversed basin and has not switched, and
#: quoting its cost against an optimum that reverses exactly compares two different things.
_SWITCHED_INFIDELITY = 1e-3


@dataclass(frozen=True)
class ConstrainedResult:
    """The outcome of a constrained pulse optimization.

    Attributes:
        times: the time grid, s.
        field: the optimized field at each time, shape ``(N, 3)``, T.
        cost: the switching cost, T^2 s.
        infidelity: the reversal infidelity ``(1 + s_z(T)) / 2``, zero for a complete reversal,
            measured by the norm-preserving RK4 integration.
        switched: whether the moment actually reversed, ``infidelity <= 1e-3``.
        final_sz: the final z-component, from the norm-preserving RK4 integration.
        peak_amplitude: the largest field magnitude used, T.
        iterations: the optimizer iterations taken.
        converged: whether the optimizer reported convergence rather than hitting its cap.
    """

    times: np.ndarray
    field: np.ndarray
    cost: float
    infidelity: float
    switched: bool
    final_sz: float
    peak_amplitude: float
    iterations: int = 0
    converged: bool = False


def _integrate_and_score(
    system: MacrospinSystem,
    field_table: np.ndarray,
    times: np.ndarray,
) -> tuple[float, float]:
    """Integrate the reversal under a tabulated field and return ``(cost, final_sz)``."""

    trajectory = integrate_llg_tabulated(np.array([0.0, 0.0, 1.0]), field_table, times, system)
    return switching_cost(times, field_table), float(trajectory[-1, 2])


class GRAPESolver:
    """Optimize a piecewise-constant transverse field under an amplitude cap and a slew penalty.

    Args:
        system: the macrospin.
        switching_time: ``T`` in s.
        n_slices: the number of piecewise-constant slices.
        amplitude_cap: the maximum field magnitude per component, T. The box constraint that makes the
            pulse realizable.
        slew_weight: the penalty weight on the squared slice-to-slice change, discouraging pulses that
            jump faster than a generator can follow. Dimensionless relative to the cost scale.
        fidelity_weight: the penalty weight on an incomplete reversal.
        integration_steps: the grid used to integrate the equation of motion. 2400 steps keep the
            adjoint's forward Euler pass and the reporting RK4 pass within 4e-5 of each other in the
            final ``s_z``, which is well inside the reversal threshold.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        switching_time: float,
        n_slices: int = 24,
        amplitude_cap: float | None = None,
        slew_weight: float = 0.0,
        fidelity_weight: float | None = None,
        integration_steps: int = 2400,
    ) -> None:
        self.system = system
        self.switching_time = switching_time
        self.n_slices = n_slices
        self.amplitude_cap = amplitude_cap if amplitude_cap is not None else 20.0 * system.anisotropy_field
        self.slew_weight = slew_weight
        # A default fidelity weight scaled to the cost of a free-macrospin reversal, so the penalty and
        # the cost are the same order of magnitude regardless of the material and switching time.
        from ..analytic.uniaxial import cost_free_macrospin

        free = cost_free_macrospin(switching_time, system.alpha, system.gamma)
        self.fidelity_weight = fidelity_weight if fidelity_weight is not None else 50.0 * free
        self.integration_steps = integration_steps
        self._slice_times = np.linspace(0.0, switching_time, n_slices)
        self._grid = np.linspace(0.0, switching_time, integration_steps)

    def _problem(self) -> LinearControlProblem:
        """The linear-control problem: slice values interpolated onto the integration grid."""
        from ..analytic.uniaxial import cost_free_macrospin

        design = interpolation_design(self._slice_times, self._grid)
        return LinearControlProblem(
            system=self.system,
            times=self._grid,
            design=design,
            fidelity_weight=self.fidelity_weight,
            field_scale=self.amplitude_cap,
            objective_scale=cost_free_macrospin(
                self.switching_time, self.system.alpha, self.system.gamma
            ),
        )

    def _field_table(self, params: np.ndarray) -> np.ndarray:
        """The (N, 3) field on the integration grid from slice parameters in tesla."""
        design = interpolation_design(self._slice_times, self._grid)
        blocks = np.asarray(params, dtype=float).reshape(2, self.n_slices)
        transverse = design @ blocks.T
        return np.concatenate([transverse, np.zeros((self._grid.size, 1))], axis=1)

    def solve(self, seed: int = 0, max_iterations: int = 200) -> ConstrainedResult:
        """Optimize the pulse on the exact adjoint gradient.

        The parameters are the slice values in units of the amplitude cap, so the box constraint is
        simply [-1, 1] on every parameter. Interpolation between nodes is a convex combination, so
        bounding the nodes bounds the pulse: the cap is enforced exactly, not penalized.

        Args:
            seed: seed for the initial random slice values.
            max_iterations: the optimizer iteration cap.

        Returns:
            The :class:`ConstrainedResult`, with the pulse re-evaluated by the norm-preserving RK4
            integration that the rest of the package uses.
        """
        problem = self._problem()
        rng = np.random.default_rng(seed)
        initial = rng.normal(scale=0.3, size=2 * self.n_slices)
        initial = np.clip(initial, -1.0, 1.0)
        bounds = [(-1.0, 1.0)] * (2 * self.n_slices)
        if self.slew_weight > 0.0:
            solution = self._solve_with_slew(problem, initial, bounds, max_iterations)
        else:
            solution = problem.solve(initial, max_iterations=max_iterations, bounds=bounds)
        return self._result(solution.field, solution.iterations, solution.converged)

    def _solve_with_slew(self, problem, initial, bounds, max_iterations):
        """The same optimization with the slew penalty, whose gradient is also exact.

        The penalty is a quadratic form in the parameters, ``sum (p_{i+1} - p_i)^2`` per block, so its
        gradient is the discrete second difference. It stays out of the adjoint entirely: the control
        enters the dynamics only through the field.
        """
        from scipy.optimize import minimize

        weight = self.slew_weight

        def penalized(vector: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = problem.objective_and_gradient(vector)
            blocks = vector.reshape(2, self.n_slices)
            differences = np.diff(blocks, axis=1)
            value += weight * float(np.sum(differences**2))
            slew_gradient = np.zeros_like(blocks)
            slew_gradient[:, :-1] -= 2.0 * weight * differences
            slew_gradient[:, 1:] += 2.0 * weight * differences
            return value, gradient + slew_gradient.reshape(-1)

        result = minimize(
            penalized,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        from .linear_basis import LinearControlSolution

        return LinearControlSolution(
            parameters=result.x,
            field=problem.field_of(result.x),
            objective=float(result.fun),
            iterations=int(result.nit),
            converged=bool(result.success),
        )

    def _result(self, field_table: np.ndarray, iterations: int, converged: bool) -> ConstrainedResult:
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        return ConstrainedResult(
            times=self._grid,
            field=field_table,
            cost=cost,
            infidelity=0.5 * (1.0 + final_sz),
            switched=0.5 * (1.0 + final_sz) <= _SWITCHED_INFIDELITY,
            final_sz=final_sz,
            peak_amplitude=float(np.max(np.linalg.norm(field_table, axis=-1))),
            iterations=iterations,
            converged=converged,
        )


class CRABSolver:
    """Optimize a band-limited transverse field in a truncated randomized Fourier basis.

    The field on each transverse component is a sum of ``n_harmonics`` sine terms whose base
    frequencies are the first harmonics of the switching window, each scaled by an optimized amplitude
    and randomized slightly, so the pulse is smooth and its bandwidth is bounded by the highest
    harmonic. That bound **is** the realizability constraint, imposed by construction rather than by a
    penalty.

    Args:
        system: the macrospin.
        switching_time: ``T`` in s.
        n_harmonics: the number of Fourier components per transverse axis. More harmonics means more
            bandwidth and a lower achievable cost, so sweeping it traces the price of realizability.
        fidelity_weight: the penalty weight on an incomplete reversal.
        integration_steps: the integration grid size; see GRAPE for why it is 2400.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        switching_time: float,
        n_harmonics: int = 6,
        fidelity_weight: float | None = None,
        integration_steps: int = 2400,
    ) -> None:
        self.system = system
        self.switching_time = switching_time
        self.n_harmonics = n_harmonics
        from ..analytic.uniaxial import cost_free_macrospin

        free = cost_free_macrospin(switching_time, system.alpha, system.gamma)
        self.fidelity_weight = fidelity_weight if fidelity_weight is not None else 50.0 * free
        self.integration_steps = integration_steps
        self._grid = np.linspace(0.0, switching_time, integration_steps)

    def bandwidth_hz(self) -> float:
        """The bandwidth ceiling of the basis, in hertz: the highest harmonic frequency used."""
        return self.n_harmonics / self.switching_time

    def _frequencies(self, seed: int) -> np.ndarray:
        """The base frequencies: the first harmonics of the window, nudged randomly.

        The nudge is the "chopped random basis": it lets the optimizer reach pulses a fixed harmonic
        grid could not, and it is fixed before the optimization so the basis stays linear.
        """
        rng = np.random.default_rng(seed)
        base = np.arange(1, self.n_harmonics + 1) / self.switching_time
        return base * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=self.n_harmonics))

    def _field_table(self, params: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """The (N, 3) field of a set of harmonic amplitudes in tesla.

        The basis carries a sine and a cosine per harmonic, so a block is twice the harmonic count.
        """
        design = harmonic_design(frequencies, self._grid, self.switching_time)
        blocks = np.asarray(params, dtype=float).reshape(2, design.shape[1])
        transverse = design @ blocks.T
        return np.concatenate([transverse, np.zeros((self._grid.size, 1))], axis=1)

    def solve(self, seed: int = 0, max_iterations: int = 300) -> ConstrainedResult:
        """Optimize the band-limited pulse on the exact adjoint gradient.

        Args:
            seed: seed for the randomized base frequencies and the initial amplitudes.
            max_iterations: the optimizer iteration cap.

        Returns:
            The :class:`ConstrainedResult`, re-evaluated with the norm-preserving RK4 integration.
        """
        from ..analytic.uniaxial import cost_free_macrospin

        frequencies = self._frequencies(seed)
        field_scale = self.system.anisotropy_field
        design = harmonic_design(frequencies, self._grid, self.switching_time)
        problem = LinearControlProblem(
            system=self.system,
            times=self._grid,
            design=design,
            fidelity_weight=self.fidelity_weight,
            field_scale=field_scale,
            objective_scale=cost_free_macrospin(
                self.switching_time, self.system.alpha, self.system.gamma
            ),
        )
        rng = np.random.default_rng(seed + 1)
        initial = rng.normal(scale=2.0, size=2 * design.shape[1])
        solution = problem.solve(initial, max_iterations=max_iterations)

        field_table = solution.field
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        return ConstrainedResult(
            times=self._grid,
            field=field_table,
            cost=cost,
            infidelity=0.5 * (1.0 + final_sz),
            switched=0.5 * (1.0 + final_sz) <= _SWITCHED_INFIDELITY,
            final_sz=final_sz,
            peak_amplitude=float(np.max(np.linalg.norm(field_table, axis=-1))),
            iterations=solution.iterations,
            converged=solution.converged,
        )
