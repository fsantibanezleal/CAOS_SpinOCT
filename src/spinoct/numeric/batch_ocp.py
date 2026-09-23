"""Many optimal control paths at once, on the GPU, with the exact gradient.

Why a batch lane exists
-----------------------
One optimal control path is a small problem: a chain of a few hundred unit vectors, minimized in a
second or two on one CPU core. A GPU does nothing for it. What the sweeps in front of this solver
actually do is solve *hundreds* of such problems that differ only in their parameters: a cost front per
material, a map over damping and hard-axis ratio, a seed sweep over coexisting optimal paths. Those are
independent, identically shaped, and embarrassingly parallel, and that is the shape a GPU is for.

This module solves a whole batch as one tensor of shape ``(B, Q + 2, 3)``. The physics is the same as
:mod:`spinoct.numeric.image_ocp`, term for term:

    s_{p+1/2}     = (s_{p+1} + s_p) / |s_{p+1} + s_p|
    s_dot_{p+1/2} = (delta_p / dt) (s_{p+1} - s_p) / |s_{p+1} - s_p|
    b             = (alpha / gamma) s_dot + (1 / gamma) [s x s_dot] - b_i_perp(s)
    Phi           = dt sum_p |b_{p+1/2}|^2

with two differences, both deliberate:

**The gradient is exact.** The CPU solver differences the cost locally with a step of 1e-7 per
component, which is what makes its iteration O(Q) instead of O(Q^2). Automatic differentiation gives
the same derivative without the truncation, at the cost of one backward pass, and on a batch that pass
is as parallel as the forward one. The two gradients agree to the finite-difference floor, and
``tests/test_batch_ocp.py`` holds them to it.

**The switching time may differ per problem, the image count may not.** A batch is one tensor, so every
problem in it is discretized with the same ``Q``; ``recommended_images`` takes the largest count the
batch asks for, which over-resolves the easy problems rather than under-resolving the hard ones.

torch is an optional dependency
-------------------------------
Install it with ``pip install spinoct[torch]``. The package never imports torch at module import time,
nothing in the default install depends on it, and CI does not install it (a GPU lane is verified once,
on a machine that has one, not on every push). Without torch this module raises with that instruction
and the CPU solver remains the reference implementation for every result this package publishes.

Which lane produced a number
----------------------------
:class:`BatchOCPResult` records the device and the engine that produced it, because "the GPU said so"
is not provenance. The acceptance gate for this lane is that its costs match the CPU solver's on the
same problems, and the closed form where one exists.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dynamics.system import MacrospinSystem
from .image_ocp import ImageOCPSolver, _geodesic_interpolate, _normalize

__all__ = ["BatchImageOCPSolver", "BatchOCPResult", "torch_is_available"]

#: Per-problem convergence: the largest tangent-gradient magnitude over the chain, divided by the
#: problem's own cost, so the test is dimensionless and the same at every switching time. Measured on
#: the CPU solver's own paths at Q = 80, uniaxial and biaxial, T from 1 to 100 tau0: the five that the
#: reference itself calls converged sit between 3.7e-06 and 5.5e-05, and the one that hits its
#: iteration cap sits at 2.2e-03. So 1e-4 is above every converged path of the reference lane and more
#: than an order of magnitude below a stalled one. A problem past it is reported unconverged rather
#: than silently averaged into a sweep.
_RESIDUAL_TOLERANCE = 1e-4
#: Guard for the zero-chord case: two identical images give a zero chord, whose normalization and whose
#: derivative are both undefined. The angle is zero there, so the velocity is zero whatever direction is
#: chosen; clamping the denominator keeps the forward value and the gradient finite instead of NaN.
_TINY_CHORD = 1e-30


def torch_is_available() -> bool:
    """Whether the optional GPU lane can run at all."""
    try:  # pragma: no cover - a one-line import probe
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "the batch lane needs PyTorch, which is an optional dependency: "
            "pip install spinoct[torch]. The CPU solver spinoct.numeric.ImageOCPSolver "
            "computes the same thing without it."
        ) from exc
    return torch


@dataclass(frozen=True)
class BatchOCPResult:
    """The outcome of a batched solve, one row per problem.

    Attributes:
        images: the converged chains, shape ``(B, Q + 2, 3)``, endpoints included.
        times: the image times of each problem, shape ``(B, Q + 2)``, s.
        costs: the switching cost of each converged path, shape ``(B,)``, T^2 s.
        converged: shape ``(B,)``, whether that problem's own residual met the tolerance. A batch is
            minimized jointly, so some problems can converge while others do not; this is per problem
            for that reason.
        residuals: shape ``(B,)``, the dimensionless residual each verdict was taken from.
        seeds: shape ``(B,)``, the seed whose solve is reported (``solve_best`` keeps the cheapest).
        iterations: L-BFGS iterations spent on the batch.
        device: the device the minimization ran on, ``"cuda"`` or ``"cpu"``.
        engine: the engine and version that produced the numbers, for provenance.
        seconds: wall-clock time of the solve, s.
    """

    images: np.ndarray
    times: np.ndarray
    costs: np.ndarray
    converged: np.ndarray
    residuals: np.ndarray
    seeds: np.ndarray
    iterations: int
    device: str
    engine: str
    seconds: float

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))


class BatchImageOCPSolver:
    """Solve a batch of independent optimal control paths as one tensor.

    Args:
        systems: the macrospins, one per problem. They may differ in every parameter.
        switching_times: the switching time of each problem, s. Either one value per system or a
            single value used for all of them.
        n_images: the interior image count ``Q``, shared by the batch. Use
            :meth:`recommended_images`, which takes the largest count any problem in the batch needs.
        device: ``"cuda"``, ``"cpu"``, or ``None`` to take a GPU when one is present.
        dtype: ``"float64"`` (default) or ``"float32"``. Double precision is the default because the
            cost is a sum of squares of fields that differ by orders of magnitude across a sweep, and
            the acceptance gate against the CPU solver is tighter than float32 can hold.
    """

    def __init__(
        self,
        systems: Sequence[MacrospinSystem],
        switching_times: Sequence[float] | float,
        n_images: int,
        device: str | None = None,
        dtype: str = "float64",
    ) -> None:
        torch = _require_torch()
        if len(systems) < 1:
            raise ValueError("a batch needs at least one system")
        if n_images < 1:
            raise ValueError("n_images (Q) must be at least 1")
        times = np.atleast_1d(np.asarray(switching_times, dtype=float))
        if times.size == 1:
            times = np.repeat(times, len(systems))
        if times.size != len(systems):
            raise ValueError("switching_times must have one entry per system, or be a single value")
        if np.any(times <= 0.0):
            raise ValueError("every switching_time must be positive (s)")

        self.systems = list(systems)
        self.switching_times = times
        self.n_images = int(n_images)
        self.batch_size = len(systems)
        self.times = np.stack([np.linspace(0.0, t, n_images + 2) for t in times])
        self._dt = times / (n_images + 1)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was asked for and no CUDA device is visible")
        self.device = device
        self._torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        if dtype not in ("float64", "float32"):
            raise ValueError("dtype must be 'float64' or 'float32'")

        def column(values: list[float]) -> Any:
            return torch.tensor(values, dtype=self._torch_dtype, device=device).reshape(-1, 1, 1)

        self._alpha = column([s.alpha for s in systems])
        self._gamma = column([s.gamma for s in systems])
        self._xi = column([s.hard_axis_ratio for s in systems])
        # The internal field is b_i = (2K/mu) (-xi s_x, 0, s_z); this is the 2K/mu in front.
        self._field_scale = column([2.0 * s.anisotropy_j / s.mu for s in systems])
        self._dt_t = column(list(self._dt))

    # ------------------------------------------------------------------ resolution

    @staticmethod
    def recommended_images(
        systems: Sequence[MacrospinSystem], switching_times: Sequence[float] | float
    ) -> int:
        """The largest image count any problem in the batch would ask for on its own.

        One tensor means one discretization. Taking the maximum over-resolves the easy problems, which
        costs time; taking anything less under-resolves the hard ones, which costs correctness.
        """
        times = np.atleast_1d(np.asarray(switching_times, dtype=float))
        if times.size == 1:
            times = np.repeat(times, len(systems))
        return int(
            max(
                ImageOCPSolver.recommended_images(system, float(time))
                for system, time in zip(systems, times, strict=True)
            )
        )

    # ------------------------------------------------------------------ the cost

    def _chain_from(self, interior: Any) -> Any:
        """Clamp the endpoints and put the interior on the sphere: ``s = x / |x|``."""
        torch = _require_torch()
        normalized = interior / torch.linalg.norm(interior, dim=-1, keepdim=True)
        poles = torch.zeros(
            (self.batch_size, 1, 3), dtype=self._torch_dtype, device=self.device
        )
        start = poles.clone()
        start[:, 0, 2] = 1.0
        end = poles.clone()
        end[:, 0, 2] = -1.0
        return torch.cat([start, normalized, end], dim=1)

    def _costs_tensor(self, chain: Any) -> Any:
        """The discretized cost of every chain in the batch, shape ``(B,)``, T^2 s."""
        torch = _require_torch()
        left = chain[:, :-1]
        right = chain[:, 1:]
        total = left + right
        chord = right - left
        total_norm = torch.linalg.norm(total, dim=-1, keepdim=True)
        chord_norm = torch.linalg.norm(chord, dim=-1, keepdim=True)
        midpoints = total / total_norm
        # Geodesic angle, exact at every angle (arccos of the dot loses precision for close images).
        delta = 2.0 * torch.atan2(chord_norm, total_norm)
        direction = chord / chord_norm.clamp_min(_TINY_CHORD)
        velocity = (delta / self._dt_t) * direction

        internal = torch.stack(
            [
                -self._field_scale[..., 0] * self._xi[..., 0] * midpoints[..., 0],
                torch.zeros_like(midpoints[..., 1]),
                self._field_scale[..., 0] * midpoints[..., 2],
            ],
            dim=-1,
        )
        longitudinal = torch.sum(internal * midpoints, dim=-1, keepdim=True)
        internal_perp = internal - longitudinal * midpoints

        fields = (
            (self._alpha / self._gamma) * velocity
            + torch.cross(midpoints, velocity, dim=-1) / self._gamma
            - internal_perp
        )
        return torch.sum(fields**2, dim=(-1, -2)) * self._dt_t[:, 0, 0]

    def cost_of(self, images: np.ndarray) -> np.ndarray:
        """The cost of a batch of chains, shape ``(B, Q + 2, 3)`` in, shape ``(B,)`` out, T^2 s."""
        torch = _require_torch()
        chain = torch.as_tensor(np.asarray(images, dtype=float), dtype=self._torch_dtype, device=self.device)
        with torch.no_grad():
            return self._costs_tensor(chain).detach().cpu().numpy()

    # ------------------------------------------------------------------ the solve

    def _initial_interior(self, seed: int, noise: float) -> np.ndarray:
        """The perturbed great-circle chain each problem starts from, shape ``(B, Q, 3)``.

        Built with NumPy and the same generator sequence as the CPU solver, so a batch of one is the
        same problem from the same start as ``ImageOCPSolver.solve(seed=...)`` and the two lanes can be
        compared path by path rather than only in distribution.
        """
        chains = []
        for index in range(self.batch_size):
            rng = np.random.default_rng(seed)
            chain = _geodesic_interpolate(
                np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), self.n_images + 2
            )
            if noise > 0.0:
                perturbation = rng.normal(scale=noise, size=(self.n_images, 3))
                chain[1:-1] = _normalize(chain[1:-1] + perturbation)
            chains.append(chain[1:-1])
            del index
        return np.stack(chains)

    def solve(
        self,
        max_iterations: int = 600,
        seed: int = 0,
        noise: float = 0.05,
        relative_tolerance: float = 1e-12,
        residual_tolerance: float = _RESIDUAL_TOLERANCE,
    ) -> BatchOCPResult:
        """Minimize every problem in the batch at once.

        The objective handed to L-BFGS is the sum of the per-problem costs, each divided by its own
        starting cost. The sum is separable, so its minimizer is each problem's minimizer; the
        normalization is what keeps one expensive problem in the batch from setting the step size for
        all of them, and it is the same rescaling the CPU solver applies to a single problem.

        Args:
            max_iterations: the L-BFGS iteration cap for the batch.
            seed: the symmetry-breaking seed. The pole-to-pole geodesic sits in a symmetry plane where
                the gradient vanishes to first order, so a solve started exactly on it reports the
                meridian as converged. Breaking the symmetry is a correctness requirement, not a
                tuning choice.
            noise: the standard deviation, in radians of arc, of that perturbation.
            relative_tolerance: L-BFGS termination on the objective.
            residual_tolerance: the per-problem convergence test, dimensionless.
        """
        torch = _require_torch()
        started = time.perf_counter()

        interior = torch.tensor(
            self._initial_interior(seed, noise), dtype=self._torch_dtype, device=self.device
        ).requires_grad_(True)
        with torch.no_grad():
            tiny = torch.finfo(self._torch_dtype).tiny
            scale = self._costs_tensor(self._chain_from(interior)).clamp_min(tiny)

        optimizer = torch.optim.LBFGS(
            [interior],
            max_iter=max_iterations,
            tolerance_change=relative_tolerance,
            tolerance_grad=1e-14,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        iterations = {"n": 0}

        def closure() -> Any:
            optimizer.zero_grad(set_to_none=True)
            objective = torch.sum(self._costs_tensor(self._chain_from(interior)) / scale)
            objective.backward()
            iterations["n"] += 1
            return objective

        optimizer.step(closure)

        chain = self._chain_from(interior)
        costs = self._costs_tensor(chain)
        # The residual is measured on the sphere, not on the unconstrained x: the component of the
        # gradient along an image does nothing to the path, and only the tangent part is a force.
        gradient = torch.autograd.grad(torch.sum(costs), interior, retain_graph=False)[0]
        with torch.no_grad():
            unit = interior / torch.linalg.norm(interior, dim=-1, keepdim=True)
            tangent = gradient - torch.sum(gradient * unit, dim=-1, keepdim=True) * unit
            residuals = torch.max(torch.linalg.norm(tangent, dim=-1), dim=-1).values / costs
            images = chain.detach().cpu().numpy()
            cost_values = costs.detach().cpu().numpy()
            residual_values = residuals.detach().cpu().numpy()

        if self.device == "cuda":
            torch.cuda.synchronize()
        return BatchOCPResult(
            images=images,
            times=self.times,
            costs=cost_values,
            converged=residual_values <= residual_tolerance,
            residuals=residual_values,
            seeds=np.full(self.batch_size, seed, dtype=int),
            iterations=iterations["n"],
            device=self.device,
            engine=f"torch {torch.__version__}",
            seconds=time.perf_counter() - started,
        )

    def solve_best(
        self,
        n_seeds: int = 8,
        base_seed: int = 0,
        noise: float = 0.15,
        **solve_kwargs: Any,
    ) -> BatchOCPResult:
        """Solve every problem from several seeds and keep each problem's cheapest path.

        This is where the lane earns its existence. Multiple optimal control paths coexist in the
        biaxial case, so a single seed reports whichever basin it landed in; the CPU answer to that is
        a loop over seeds, one solve at a time. Here the seeds are more rows of the same tensor, so a
        sweep of ``B`` problems over ``S`` seeds is one minimization of ``B * S`` rows.
        """
        results = [
            self.solve(seed=base_seed + offset, noise=noise, **solve_kwargs) for offset in range(n_seeds)
        ]
        costs = np.stack([r.costs for r in results])
        winner = np.argmin(costs, axis=0)
        rows = np.arange(self.batch_size)
        return BatchOCPResult(
            images=np.stack([results[w].images[i] for i, w in zip(rows, winner, strict=True)]),
            times=self.times,
            costs=costs[winner, rows],
            converged=np.stack([r.converged for r in results])[winner, rows],
            residuals=np.stack([r.residuals for r in results])[winner, rows],
            seeds=np.array([base_seed + int(w) for w in winner], dtype=int),
            iterations=sum(r.iterations for r in results),
            device=self.device,
            engine=results[0].engine,
            seconds=sum(r.seconds for r in results),
        )
