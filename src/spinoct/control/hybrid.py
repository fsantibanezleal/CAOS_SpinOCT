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
from scipy.optimize import minimize

from ..dynamics.llg import switching_cost
from ..dynamics.system import MacrospinSystem

__all__ = ["HybridResult", "HybridSolver", "integrate_llg_sot"]

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

    def _tables(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the field and current tables from the Fourier coefficients."""
        n = self.n_harmonics
        envelope = np.sin(np.pi * self._grid / self.switching_time)
        field = np.zeros((self._grid.size, 3))
        current = np.zeros((self._grid.size, 3))
        idx = 0
        for comp in (0, 1):  # transverse x, y for both field and current
            for h in range(1, n + 1):
                phase = np.sin(2.0 * np.pi * h * self._grid / self.switching_time)
                field[:, comp] += params[idx] * phase * envelope
                idx += 1
                current[:, comp] += params[idx] * phase * envelope
                idx += 1
        return field, current

    def _objective(self, params: np.ndarray) -> float:
        field, current = self._tables(params)
        trajectory = integrate_llg_sot(
            np.array([0.0, 0.0, 1.0]), field, current, self._grid, self.system, self.xi_f, self.xi_d
        )
        final_sz = float(trajectory[-1, 2])
        field_cost = switching_cost(self._grid, field)
        current_cost = switching_cost(self._grid, current)
        weighted = self.circuit_field * field_cost + self.circuit_current * current_cost
        infidelity = 0.5 * (1.0 + final_sz)
        return weighted + 50.0 * self._free * infidelity

    def solve(self, seed: int = 0, max_iterations: int = 200) -> HybridResult:
        """Co-optimize the field and the current.

        Args:
            seed: initial-guess seed.
            max_iterations: optimizer iteration cap.

        Returns:
            The :class:`HybridResult`.
        """
        rng = np.random.default_rng(seed)
        n_params = 4 * self.n_harmonics  # 2 controls x 2 components x n harmonics
        scale = 0.5 * self.system.anisotropy_field
        initial = rng.normal(scale=scale, size=n_params)
        result = minimize(
            self._objective,
            initial,
            method="Nelder-Mead",
            options={"maxiter": max_iterations * n_params, "xatol": 1e-9, "fatol": 1e-30},
        )
        field, current = self._tables(result.x)
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
            switched=final_sz < 0.0,
            final_sz=final_sz,
        )
