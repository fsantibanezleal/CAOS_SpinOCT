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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..dynamics.llg import integrate_llg_tabulated, switching_cost
from ..dynamics.system import MacrospinSystem

__all__ = ["ConstrainedResult", "CRABSolver", "GRAPESolver"]


@dataclass(frozen=True)
class ConstrainedResult:
    """The outcome of a constrained pulse optimization.

    Attributes:
        times: the time grid, s.
        field: the optimized field at each time, shape ``(N, 3)``, T.
        cost: the switching cost, T^2 s.
        infidelity: the reversal infidelity ``(1 + s_z(T)) / 2``, zero for a complete reversal.
        switched: whether the moment ended in the reversed basin.
        final_sz: the final z-component.
        peak_amplitude: the largest field magnitude used, T.
    """

    times: np.ndarray
    field: np.ndarray
    cost: float
    infidelity: float
    switched: bool
    final_sz: float
    peak_amplitude: float


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
        integration_steps: the grid used to integrate the equation of motion during optimization.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        switching_time: float,
        n_slices: int = 24,
        amplitude_cap: float | None = None,
        slew_weight: float = 0.0,
        fidelity_weight: float | None = None,
        integration_steps: int = 1200,
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

    def _field_table(self, params: np.ndarray) -> np.ndarray:
        """Build the (N, 3) field on the integration grid from the slice parameters (bx, by)."""
        bx = np.interp(self._grid, self._slice_times, params[: self.n_slices])
        by = np.interp(self._grid, self._slice_times, params[self.n_slices :])
        return np.stack([bx, by, np.zeros_like(bx)], axis=-1)

    def _objective(self, params: np.ndarray) -> float:
        field_table = self._field_table(params)
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        infidelity = 0.5 * (1.0 + final_sz)
        penalty = self.fidelity_weight * infidelity
        if self.slew_weight > 0.0:
            slew = np.sum(np.diff(params[: self.n_slices]) ** 2) + np.sum(
                np.diff(params[self.n_slices :]) ** 2
            )
            penalty += self.slew_weight * slew
        return cost + penalty

    def solve(self, seed: int = 0, max_iterations: int = 200) -> ConstrainedResult:
        """Optimize the pulse.

        Args:
            seed: seed for the initial random slice values.
            max_iterations: the optimizer iteration cap.

        Returns:
            The :class:`ConstrainedResult`, with the pulse re-evaluated on the integration grid.
        """
        rng = np.random.default_rng(seed)
        scale = 0.3 * self.amplitude_cap
        initial = rng.normal(scale=scale, size=2 * self.n_slices)
        bounds = [(-self.amplitude_cap, self.amplitude_cap)] * (2 * self.n_slices)
        result = minimize(
            self._objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        field_table = self._field_table(result.x)
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        return ConstrainedResult(
            times=self._grid,
            field=field_table,
            cost=cost,
            infidelity=0.5 * (1.0 + final_sz),
            switched=final_sz < 0.0,
            final_sz=final_sz,
            peak_amplitude=float(np.max(np.linalg.norm(field_table, axis=-1))),
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
        integration_steps: the integration grid size.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        switching_time: float,
        n_harmonics: int = 6,
        fidelity_weight: float | None = None,
        integration_steps: int = 1200,
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

    def _field_table(self, params: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        bx = np.zeros_like(self._grid)
        by = np.zeros_like(self._grid)
        amps_x = params[: self.n_harmonics]
        amps_y = params[self.n_harmonics :]
        for harmonic in range(self.n_harmonics):
            phase = 2.0 * np.pi * frequencies[harmonic] * self._grid
            # A window that vanishes at both ends keeps the pulse from starting or ending abruptly,
            # which a real generator would round off anyway.
            envelope = np.sin(np.pi * self._grid / self.switching_time)
            bx += amps_x[harmonic] * np.sin(phase) * envelope
            by += amps_y[harmonic] * np.sin(phase) * envelope
        return np.stack([bx, by, np.zeros_like(bx)], axis=-1)

    def _objective(self, params: np.ndarray, frequencies: np.ndarray) -> float:
        field_table = self._field_table(params, frequencies)
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        return cost + self.fidelity_weight * 0.5 * (1.0 + final_sz)

    def solve(self, seed: int = 0, max_iterations: int = 300) -> ConstrainedResult:
        """Optimize the band-limited pulse.

        Args:
            seed: seed for the randomized base frequencies and the initial amplitudes.
            max_iterations: the optimizer iteration cap.

        Returns:
            The :class:`ConstrainedResult`.
        """
        rng = np.random.default_rng(seed)
        # Base frequencies are the first harmonics of the window, nudged randomly (the "chopped random
        # basis"), which lets the optimizer reach pulses a fixed harmonic grid could not.
        base = np.arange(1, self.n_harmonics + 1) / self.switching_time
        frequencies = base * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=self.n_harmonics))
        scale = 2.0 * self.system.anisotropy_field
        initial = rng.normal(scale=scale, size=2 * self.n_harmonics)
        result = minimize(
            lambda params: self._objective(params, frequencies),
            initial,
            method="Nelder-Mead",
            options={"maxiter": max_iterations * (2 * self.n_harmonics), "xatol": 1e-9, "fatol": 1e-30},
        )
        field_table = self._field_table(result.x, frequencies)
        cost, final_sz = _integrate_and_score(self.system, field_table, self._grid)
        return ConstrainedResult(
            times=self._grid,
            field=field_table,
            cost=cost,
            infidelity=0.5 * (1.0 + final_sz),
            switched=final_sz < 0.0,
            final_sz=final_sz,
            peak_amplitude=float(np.max(np.linalg.norm(field_table, axis=-1))),
        )
