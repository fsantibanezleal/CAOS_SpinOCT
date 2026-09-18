"""Exact gradient-based pulse optimization by the discrete adjoint method (R10).

The finite-difference gradient of the constrained solvers costs one forward integration per control
parameter. The adjoint method computes the exact gradient with respect to every control parameter in a
single backward pass, at the cost of one extra integration, independent of the number of parameters.
For a pulse with many slices this is the difference between a solver that scales and one that does not,
and it is the classical route to gradient-based optimal control.

We differentiate through a norm-projected forward Euler integration of the Landau-Lifshitz-Gilbert
equation analytically (a hand-derived reverse-mode pass), so the gradient is exact for the discretized
system and needs no automatic-differentiation dependency; the core stays pure numpy. The same reverse
pass runs elementwise, so it batches and ports to a GPU tensor library unchanged, which is the
``[torch]`` extra.

Forward step, with control field ``b_k`` piecewise constant on ``N`` slices:

    r_k = s_k + dt * f(s_k, b_k),    s_{k+1} = r_k / |r_k|,

with ``f`` the LLG right-hand side. The objective is the switching cost plus a reversal-fidelity
penalty:

    C = sum_k |b_k|^2 dt  +  lambda (1 + s_{N,z}) / 2.

The adjoint recursion propagates the cost sensitivity ``g_k = dC/ds_k`` backward and reads off
``dC/db_k`` at each step. All Jacobians are analytic because the LLG right-hand side is a sum of cross
products, which are linear in each argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics.sot_torque import explicit_sot_coefficients
from .dynamics.system import MacrospinSystem

__all__ = [
    "AdjointResult",
    "adjoint_gradient",
    "adjoint_gradient_sot",
    "optimize_pulse_adjoint",
]


def _cross_matrix(v: np.ndarray) -> np.ndarray:
    """The skew matrix ``[v]_x`` such that ``[v]_x u = v x u``."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def _f(s: np.ndarray, b_applied: np.ndarray, system: MacrospinSystem) -> np.ndarray:
    """The LLG right-hand side for one moment."""
    b_total = system.internal_field(s) + b_applied
    alpha, gamma = system.alpha, system.gamma
    precession = np.cross(s, b_total)
    damping = np.cross(s, precession)
    return (-gamma * precession - alpha * gamma * damping) / (1.0 + alpha**2)


def _f_jacobians(
    s: np.ndarray, b_applied: np.ndarray, system: MacrospinSystem
) -> tuple[np.ndarray, np.ndarray]:
    """The Jacobians ``df/ds`` and ``df/db`` at a point, each a 3x3 matrix.

    The internal field is linear in s (a diagonal anisotropy matrix), so
    ``b_total = A s + b_applied`` with ``A = diag(0, 0, 2K/mu)`` for the uniaxial easy axis. The LLG
    right-hand side is a sum of cross products, each bilinear, so the Jacobians are assembled from
    skew matrices.
    """
    alpha, gamma = system.alpha, system.gamma
    scale = 1.0 / (1.0 + alpha**2)
    anisotropy = np.diag([0.0, 0.0, 2.0 * system.anisotropy_j / system.mu])
    b_total = anisotropy @ s + b_applied

    sx = _cross_matrix(s)
    btx = _cross_matrix(b_total)
    # d/ds of (s x b_total) = [s]_x (A) - [b_total]_x, since b_total depends on s through A.
    dprec_ds = sx @ anisotropy - btx
    prec = np.cross(s, b_total)
    px = _cross_matrix(prec)
    # damping = s x prec; d/ds = [s]_x dprec_ds - [prec]_x
    ddamp_ds = sx @ dprec_ds - px
    df_ds = scale * (-gamma * dprec_ds - alpha * gamma * ddamp_ds)

    # d/db_applied: b_total depends on b_applied as identity.
    dprec_db = -btx * 0.0 + sx  # d(s x b_total)/db_applied = [s]_x
    ddamp_db = sx @ dprec_db
    df_db = scale * (-gamma * dprec_db - alpha * gamma * ddamp_db)
    return df_ds, df_db


@dataclass(frozen=True)
class AdjointResult:
    """The outcome of an adjoint-gradient pulse optimization.

    Attributes:
        times: the slice times, s, shape ``(N,)``.
        field: the optimized field per slice, shape ``(N, 3)``, T.
        cost: the switching cost, T^2 s.
        final_sz: the final z-component.
        switched: whether the moment reversed.
        objective: the final penalized objective value.
        iterations: the number of gradient steps taken.
    """

    times: np.ndarray
    field: np.ndarray
    cost: float
    final_sz: float
    switched: bool
    objective: float
    iterations: int


def _forward(
    field: np.ndarray, times: np.ndarray, system: MacrospinSystem
) -> tuple[np.ndarray, np.ndarray]:
    """Forward integration; returns the trajectory and the pre-normalization vectors."""
    n = times.size
    s = np.array([0.0, 0.0, 1.0])
    trajectory = np.empty((n, 3))
    pre_norm = np.empty((n, 3))
    trajectory[0] = s
    pre_norm[0] = s
    for k in range(n - 1):
        dt = times[k + 1] - times[k]
        r = s + dt * _f(s, field[k], system)
        pre_norm[k + 1] = r
        s = r / np.linalg.norm(r)
        trajectory[k + 1] = s
    return trajectory, pre_norm


def adjoint_gradient(
    field: np.ndarray,
    times: np.ndarray,
    system: MacrospinSystem,
    fidelity_weight: float,
) -> tuple[float, float, np.ndarray]:
    """The objective and its exact gradient with respect to the per-slice field.

    Args:
        field: the field per slice, shape ``(N, 3)``, T.
        times: the slice times, s, shape ``(N,)``.
        system: the macrospin.
        fidelity_weight: the penalty weight on ``(1 + s_z(T)) / 2``.

    Returns:
        ``(objective, final_sz, gradient)`` with ``gradient`` of shape ``(N, 3)``.
    """
    n = times.size
    trajectory, pre_norm = _forward(field, times, system)
    final_sz = float(trajectory[-1, 2])

    steps = np.diff(times)
    cost = float(np.sum(np.sum(field[:-1] ** 2, axis=1) * steps))
    objective = cost + fidelity_weight * 0.5 * (1.0 + final_sz)

    grad = np.zeros_like(field)
    # Terminal adjoint: dC/ds_N from the fidelity term.
    g = np.array([0.0, 0.0, 0.5 * fidelity_weight])

    for k in range(n - 2, -1, -1):
        dt = times[k + 1] - times[k]
        # Through the normalization s_{k+1} = r / |r|.
        r = pre_norm[k + 1]
        norm = np.linalg.norm(r)
        s_next = r / norm
        d_normalize = (np.eye(3) - np.outer(s_next, s_next)) / norm  # ds_{k+1}/dr
        g_r = d_normalize.T @ g  # dC/dr_k

        df_ds, df_db = _f_jacobians(trajectory[k], field[k], system)
        # r = s_k + dt f(s_k, b_k): dr/ds_k = I + dt df_ds; dr/db_k = dt df_db.
        # Direct cost term in b_k (this slice contributes |b_k|^2 dt).
        grad[k] += 2.0 * field[k] * dt + dt * (df_db.T @ g_r)
        # Propagate to s_k.
        g = (np.eye(3) + dt * df_ds).T @ g_r

    return objective, final_sz, grad


def optimize_pulse_adjoint(
    system: MacrospinSystem,
    switching_time: float,
    n_slices: int = 60,
    fidelity_weight: float | None = None,
    learning_rate: float | None = None,
    max_iterations: int = 400,
    seed: int = 0,
) -> AdjointResult:
    """Optimize a transverse field pulse by adjoint-gradient descent.

    Args:
        system: the macrospin (uniaxial).
        switching_time: ``T`` in s.
        n_slices: the number of piecewise-constant field slices.
        fidelity_weight: the reversal penalty weight; a scaled default is used if omitted.
        learning_rate: the gradient-descent step; a scaled default is used if omitted.
        max_iterations: the descent iteration cap.
        seed: the initial-guess seed.

    Returns:
        The :class:`AdjointResult`.
    """
    from scipy.optimize import minimize

    from .analytic.uniaxial import cost_free_macrospin

    del learning_rate  # the exact gradient is fed to L-BFGS, which sets its own step
    times = np.linspace(0.0, switching_time, n_slices)
    free = cost_free_macrospin(switching_time, system.alpha, system.gamma)
    if fidelity_weight is None:
        fidelity_weight = 50.0 * free

    # Non-dimensionalize so L-BFGS works in O(1) variables: the field in units of the anisotropy
    # field, the objective in units of the free-macrospin cost. Otherwise the objective and gradient
    # are order 1e-12 and the optimizer's default tolerances are met before it takes a single step.
    field_scale = system.anisotropy_field
    objective_scale = max(free, 1e-300)

    rng = np.random.default_rng(seed)
    # Only the two transverse components per slice are optimized; the drive stays perpendicular.
    initial = rng.normal(scale=0.3, size=(n_slices, 2)).reshape(-1)

    def unpack(vector: np.ndarray) -> np.ndarray:
        field = np.zeros((n_slices, 3))
        field[:, :2] = vector.reshape(n_slices, 2) * field_scale
        return field

    def objective_and_grad(vector: np.ndarray) -> tuple[float, np.ndarray]:
        field = unpack(vector)
        objective, _final_sz, grad = adjoint_gradient(field, times, system, fidelity_weight)
        scaled_grad = grad[:, :2].reshape(-1) * field_scale / objective_scale
        return objective / objective_scale, scaled_grad

    result = minimize(
        objective_and_grad,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-9},
    )

    field = unpack(result.x)
    objective, final_sz, _ = adjoint_gradient(field, times, system, fidelity_weight)
    cost = float(np.sum(np.sum(field[:-1] ** 2, axis=1) * np.diff(times)))
    return AdjointResult(
        times=times,
        field=field,
        cost=cost,
        final_sz=final_sz,
        switched=final_sz < 0.0,
        objective=objective,
        iterations=int(result.nit),
    )


# ---------------------------------------------------------------- the spin-orbit-torque extension

#: The current-induced effective axis is the in-plane current crossed with the film normal.
_E_Z = np.array([0.0, 0.0, 1.0])


def _sot_f(
    s: np.ndarray,
    b_applied: np.ndarray,
    current: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
) -> np.ndarray:
    """The LLG right-hand side including spin-orbit torque, matching ``control.hybrid._sot_rhs``."""
    alpha, gamma = system.alpha, system.gamma
    # The couplings arrive in the source's Gilbert form; the explicit equation needs them converted.
    xi_f, xi_d = explicit_sot_coefficients(xi_f, xi_d, alpha)
    b_total = system.internal_field(s) + b_applied
    spin_hall = np.cross(current, _E_Z)
    rhs = (
        -gamma * np.cross(s, b_total)
        - alpha * gamma * np.cross(s, np.cross(s, b_total))
        + gamma * xi_f * np.cross(s, spin_hall)
        + gamma * xi_d * np.cross(s, np.cross(s, spin_hall))
    )
    return rhs / (1.0 + alpha**2)


def _sot_jacobians(
    s: np.ndarray,
    b_applied: np.ndarray,
    current: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(df/ds, df/db, df/dj)`` at a point, each 3x3.

    Every term is a cross product, so every derivative is a product of skew matrices. Writing
    ``[v]`` for the skew matrix of ``v`` and ``p = j x z`` for the current-induced axis:

        d(s x p)/ds = -[p],              d(s x p)/dp = [s]
        d(s x (s x p))/ds = -[s][p] - [s x p],   d(s x (s x p))/dp = [s][s]
        dp/dj = -[z]

    and the field terms are the ones :func:`_f_jacobians` already returns.
    """
    alpha, gamma = system.alpha, system.gamma
    xi_f, xi_d = explicit_sot_coefficients(xi_f, xi_d, alpha)
    df_ds, df_db = _f_jacobians(s, b_applied, system)

    spin_hall = np.cross(current, _E_Z)
    skew_s = _cross_matrix(s)
    skew_p = _cross_matrix(spin_hall)
    skew_sp = _cross_matrix(np.cross(s, spin_hall))

    scale = gamma / (1.0 + alpha**2)
    df_ds = df_ds + scale * (xi_f * (-skew_p) + xi_d * (-skew_s @ skew_p - skew_sp))
    # dp/dj = -[z] because p = j x z = -(z x j).
    dp_dj = -_cross_matrix(_E_Z)
    df_dj = scale * (xi_f * skew_s + xi_d * (skew_s @ skew_s)) @ dp_dj
    return df_ds, df_db, df_dj


def _sot_forward(
    field: np.ndarray,
    current: np.ndarray,
    times: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward integration of the SOT dynamics; returns the trajectory and pre-normalization vectors."""
    n = times.size
    s = np.array([0.0, 0.0, 1.0])
    trajectory = np.empty((n, 3))
    pre_norm = np.empty((n, 3))
    trajectory[0] = s
    pre_norm[0] = s
    for k in range(n - 1):
        dt = times[k + 1] - times[k]
        r = s + dt * _sot_f(s, field[k], current[k], system, xi_f, xi_d)
        pre_norm[k + 1] = r
        s = r / np.linalg.norm(r)
        trajectory[k + 1] = s
    return trajectory, pre_norm


def adjoint_gradient_sot(
    field: np.ndarray,
    current: np.ndarray,
    times: np.ndarray,
    system: MacrospinSystem,
    xi_f: float,
    xi_d: float,
    fidelity_weight: float,
    field_weight: float = 1.0,
    current_weight: float = 1.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """The two-term objective and its exact gradient with respect to BOTH controls.

    The hybrid problem prices a field against a current:

        C = C_b integral |b|^2 dt + C_j integral |j|^2 dt + lambda (1 + s_z(T)) / 2

    and the question it exists to answer is where the optimum sits as the relative price moves. One
    backward pass gives the gradient with respect to every sample of both controls at once.

    Args:
        field: the applied field per step, shape ``(N, 3)``, T.
        current: the in-plane current per step, shape ``(N, 3)``, reduced units.
        times: the grid, s, shape ``(N,)``.
        system: the macrospin.
        xi_f: field-like spin-orbit-torque coupling.
        xi_d: damping-like spin-orbit-torque coupling.
        fidelity_weight: the penalty weight on ``(1 + s_z(T)) / 2``.
        field_weight: the field-cost weight ``C_b``.
        current_weight: the current-cost weight ``C_j``.

    Returns:
        ``(objective, final_sz, grad_field, grad_current)``, the gradients of shape ``(N, 3)``.
    """
    n = times.size
    trajectory, pre_norm = _sot_forward(field, current, times, system, xi_f, xi_d)
    final_sz = float(trajectory[-1, 2])

    steps = np.diff(times)
    field_cost = float(np.sum(np.sum(field[:-1] ** 2, axis=1) * steps))
    current_cost = float(np.sum(np.sum(current[:-1] ** 2, axis=1) * steps))
    objective = (
        field_weight * field_cost
        + current_weight * current_cost
        + fidelity_weight * 0.5 * (1.0 + final_sz)
    )

    grad_field = np.zeros_like(field)
    grad_current = np.zeros_like(current)
    g = np.array([0.0, 0.0, 0.5 * fidelity_weight])

    for k in range(n - 2, -1, -1):
        dt = times[k + 1] - times[k]
        r = pre_norm[k + 1]
        norm = np.linalg.norm(r)
        s_next = r / norm
        d_normalize = (np.eye(3) - np.outer(s_next, s_next)) / norm
        g_r = d_normalize.T @ g

        df_ds, df_db, df_dj = _sot_jacobians(
            trajectory[k], field[k], current[k], system, xi_f, xi_d
        )
        grad_field[k] += 2.0 * field_weight * field[k] * dt + dt * (df_db.T @ g_r)
        grad_current[k] += 2.0 * current_weight * current[k] * dt + dt * (df_dj.T @ g_r)
        g = (np.eye(3) + dt * df_ds).T @ g_r

    return objective, final_sz, grad_field, grad_current
