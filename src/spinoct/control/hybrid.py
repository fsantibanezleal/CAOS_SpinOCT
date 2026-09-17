"""Joint field-plus-spin-orbit-torque optimal control (R13).

The kickoff paper ends by naming this design space explicitly: OCT-field implementations "could
complement or hybridize with current- and light-driven approaches ... where magnetic switching is
tailored by the interplay of fields, currents, and photons" (Badarneh, Cai, Santos, Adv. Mater. e23059,
2026). The field-only problem (Kwiatkowski et al., PRL 126, 177206, 2021) and the current-only problem
(Vlasov et al., PRB 105, 134404, 2022) are each solved; the joint problem is not.

This module co-optimizes an applied field and an in-plane electric current at once, under a two-term
cost that weighs the Joule heating of each source separately:

    Phi = C_b integral |b|^2 dt  +  C_j integral |j|^2 dt.

The two circuit constants ``C_b`` and ``C_j`` are the design knobs: their ratio sets how expensive
field is relative to current, and sweeping it traces whether a hybrid ever beats the better of the two
pure protocols. The honest prior is that it might not, and a null result is a real finding.

Dynamics
--------
The spin-orbit torque enters the Landau-Lifshitz-Gilbert equation through field-like (FL) and
damping-like (DL) terms with a current ``j`` in the plane (Vlasov et al.):

    s_dot = -gamma s x b_tot + alpha s x s_dot
            + gamma xi_F s x (j x e_z) + gamma xi_D s x [s x (j x e_z)]

with ``b_tot`` the internal plus applied field. Here ``j`` is a two-component in-plane current and the
FL/DL couplings ``xi_F``, ``xi_D`` are material constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..adjoint import adjoint_gradient_sot
from ..dynamics.llg import switching_cost
from ..dynamics.system import MacrospinSystem
from .linear_basis import harmonic_design

__all__ = ["HybridResult", "HybridSolver", "integrate_llg_sot"]

#: The reversal penalty starts at this multiple of the field cost of a free-macrospin reversal and
#: is raised until the moment reverses, the same continuation the constrained solvers use.
_FIDELITY_WEIGHT_SCALE = 50.0
_TARGET_INFIDELITY = 1e-5
_SWITCHED_INFIDELITY = 1e-3
_PENALTY_GROWTH = 10.0
_MAX_PENALTY_ROUNDS = 8

_E_Z = np.array([0.0, 0.0, 1.0])


def _sot_rhs(
    s: np.ndarray,
    b_total: np.ndarray,
    current: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
) -> np.ndarray:
    """The LLG right-hand side including spin-orbit torque, for one moment."""
    alpha, gamma = system.alpha, system.gamma
    spin_hall = np.cross(current, _E_Z)  # the current-induced effective axis, in-plane current x z
    precession = np.cross(s, b_total)
    damping = np.cross(s, precession)
    field_like = np.cross(s, spin_hall)
    damping_like = np.cross(s, np.cross(s, spin_hall))
    rhs = (
        -gamma * precession
        - alpha * gamma * damping
        + gamma * xi_f * field_like
        + gamma * xi_d * damping_like
    )
    return rhs / (1.0 + alpha**2)


def integrate_llg_sot(
    s0: np.ndarray,
    field_table: np.ndarray,
    current_table: np.ndarray,
    times: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
) -> np.ndarray:
    """Integrate the SOT-augmented equation of motion under tabulated field and current.

    Args:
        s0: initial unit moment, shape ``(3,)``.
        field_table: applied field per step, shape ``(N, 3)``, T.
        current_table: in-plane current per step, shape ``(N, 3)`` (z ignored), reduced units.
        times: the grid, s.
        system: the macrospin.
        xi_f: field-like SOT coupling.
        xi_d: damping-like SOT coupling.

    Returns:
        The trajectory, shape ``(N, 3)``, norm-preserving RK4.
    """
    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, 3))
    s = np.asarray(s0, dtype=float)
    s = s / np.linalg.norm(s)
    out[0] = s
    for i in range(times.size - 1):
        step = times[i + 1] - times[i]
        b0, b1 = field_table[i], field_table[i + 1]
        j0, j1 = current_table[i], current_table[i + 1]
        bm, jm = 0.5 * (b0 + b1), 0.5 * (j0 + j1)
        k1 = _sot_rhs(s, system.internal_field(s) + b0, j0, system, xi_f, xi_d)
        s2 = s + 0.5 * step * k1
        k2 = _sot_rhs(s2, system.internal_field(s2) + bm, jm, system, xi_f, xi_d)
        s3 = s + 0.5 * step * k2
        k3 = _sot_rhs(s3, system.internal_field(s3) + bm, jm, system, xi_f, xi_d)
        s4 = s + step * k3
        k4 = _sot_rhs(s4, system.internal_field(s4) + b1, j1, system, xi_f, xi_d)
        s = s + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = s / np.linalg.norm(s)
        out[i + 1] = s
    return out


@dataclass(frozen=True)
class HybridResult:
    """The outcome of a hybrid field-plus-current optimization.

    Attributes:
        times: the grid, s.
        field: the optimized field, shape ``(N, 3)``, T.
        current: the optimized current, shape ``(N, 3)``, reduced units.
        field_cost: ``integral |b|^2 dt``, T^2 s.
        current_cost: ``integral |j|^2 dt``, reduced units.
        weighted_cost: ``C_b field_cost + C_j current_cost``.
        field_fraction: the share of the weighted cost carried by the field.
        switched: whether the moment reversed.
        final_sz: the final z-component.
    """

    times: np.ndarray
    field: np.ndarray
    current: np.ndarray
    field_cost: float
    current_cost: float
    weighted_cost: float
    field_fraction: float
    switched: bool
    final_sz: float


class HybridSolver:
    """Co-optimize a field and an in-plane current with a two-term cost (R13).

    Both controls are band-limited (a few Fourier harmonics with a smooth envelope, as in CRAB), so
    the pulses are realizable and the optimization is over a handful of coefficients.

    Args:
        system: the macrospin (uniaxial).
        switching_time: ``T`` in s.
        circuit_field: the field-cost weight ``C_b``.
        circuit_current: the current-cost weight ``C_j``.
        xi_f: field-like SOT coupling.
        xi_d: damping-like SOT coupling.
        n_harmonics: Fourier harmonics per control component.
        integration_steps: the integration grid size.
    """

    def __init__(
        self,
        system: MacrospinSystem,
        switching_time: float,
        circuit_field: float = 1.0,
        circuit_current: float = 1.0,
        xi_f: float = 0.05,
        xi_d: float = 0.05,
        n_harmonics: int = 3,
        integration_steps: int = 300,
    ) -> None:
        self.system = system
        self.switching_time = switching_time
        self.circuit_field = circuit_field
        self.circuit_current = circuit_current
        self.xi_f = xi_f
        self.xi_d = xi_d
        self.n_harmonics = n_harmonics
        self.integration_steps = integration_steps
        self._grid = np.linspace(0.0, switching_time, integration_steps)
        from ..analytic.uniaxial import cost_free_macrospin

        self._free = cost_free_macrospin(switching_time, system.alpha, system.gamma)

    def _design(self) -> np.ndarray:
        """The shared harmonic basis for both controls, sine and cosine per harmonic.

        Both quadratures are needed for the same reason as in CRAB: with sine terms only the two
        transverse components share a phase, so neither the field nor the current can rotate, and a
        rotating drive is what reverses a moment cheaply.
        """
        frequencies = np.arange(1, self.n_harmonics + 1) / self.switching_time
        return harmonic_design(frequencies, self._grid, self.switching_time)

    def _tables(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the field (T) and current (reduced) tables from the basis coefficients.

        The parameter vector is four blocks: field x, field y, current x, current y. The field blocks
        are in units of the anisotropy field and the current blocks in the reduced units the
        spin-orbit-torque coupling is defined in, so every parameter is order one.
        """
        design = self._design()
        width = design.shape[1]
        blocks = np.asarray(params, dtype=float).reshape(4, width)
        field = np.zeros((self._grid.size, 3))
        current = np.zeros((self._grid.size, 3))
        field[:, :2] = design @ blocks[:2].T * self.system.anisotropy_field
        current[:, :2] = design @ blocks[2:].T
        return field, current

    def _objective_and_gradient(self, params: np.ndarray, fidelity_weight: float):
        """The weighted cost with its exact gradient, from one backward pass over both controls."""
        design = self._design()
        width = design.shape[1]
        field, current = self._tables(params)
        objective, final_sz, grad_field, grad_current = adjoint_gradient_sot(
            field,
            current,
            self._grid,
            self.system,
            self.xi_f,
            self.xi_d,
            fidelity_weight,
            field_weight=self.circuit_field,
            current_weight=self.circuit_current,
        )
        scale = max(self.circuit_field * self._free, 1e-300)
        grad_blocks = np.empty((4, width))
        grad_blocks[:2] = (design.T @ grad_field[:, :2]).T * self.system.anisotropy_field
        grad_blocks[2:] = (design.T @ grad_current[:, :2]).T
        return objective / scale, grad_blocks.reshape(-1) / scale, final_sz

    def solve(self, seed: int = 0, max_iterations: int = 200) -> HybridResult:
        """Co-optimize the field and the current on the exact adjoint gradient.

        The reversal penalty is raised and the solve restarted until the moment actually reverses, so a
        reported cost never belongs to a pulse that stopped half way.

        Args:
            seed: initial-guess seed.
            max_iterations: optimizer iteration cap per penalty round.

        Returns:
            The :class:`HybridResult`.
        """
        from scipy.optimize import minimize

        design = self._design()
        rng = np.random.default_rng(seed)
        vector = rng.normal(scale=0.5, size=4 * design.shape[1])
        weight = _FIDELITY_WEIGHT_SCALE * self.circuit_field * self._free
        for _round in range(_MAX_PENALTY_ROUNDS):
            result = minimize(
                lambda p, w=weight: self._objective_and_gradient(p, w)[:2],
                vector,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iterations, "ftol": 1e-14, "gtol": 1e-12},
            )
            vector = result.x
            _value, _grad, final_sz = self._objective_and_gradient(vector, weight)
            if 0.5 * (1.0 + final_sz) <= _TARGET_INFIDELITY:
                break
            weight *= _PENALTY_GROWTH

        field, current = self._tables(vector)
        trajectory = integrate_llg_sot(
            np.array([0.0, 0.0, 1.0]), field, current, self._grid, self.system, self.xi_f, self.xi_d
        )
        final_sz = float(trajectory[-1, 2])
        field_cost = switching_cost(self._grid, field)
        current_cost = switching_cost(self._grid, current)
        weighted = self.circuit_field * field_cost + self.circuit_current * current_cost
        field_share = self.circuit_field * field_cost / weighted if weighted > 0 else 0.0
        return HybridResult(
            times=self._grid,
            field=field,
            current=current,
            field_cost=field_cost,
            current_cost=current_cost,
            weighted_cost=weighted,
            field_fraction=field_share,
            switched=0.5 * (1.0 + final_sz) <= _SWITCHED_INFIDELITY,
            final_sz=final_sz,
        )
