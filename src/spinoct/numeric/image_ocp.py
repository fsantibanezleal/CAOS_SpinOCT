"""Numerical optimal control path by direct minimization of the discretized cost.

Method
------
Badarneh, M. H. A., Kwiatkowski, G. J., Bessarab, P. F., *Reduction of energy cost of magnetization
switching in a biaxial nanoparticle by use of internal dynamics*, Phys. Rev. B 107, 214448 (2023),
https://doi.org/10.1103/PhysRevB.107.214448.

The optimal control path is the trajectory that minimizes the switching cost. There is no general
closed form, so the trajectory is represented by a chain of `Q + 2` unit vectors (images) on the
sphere, the two endpoints clamped, and the cost is minimized directly over the `Q` interior images.

The discretized cost (Eq. 11) uses the midpoint rule:

    Phi = sum_p |B_{p+1/2}|^2 (t_{p+1} - t_p)

where the field at each midpoint comes from the inverted equation of motion (Eq. 4) evaluated at the
midpoint position and velocity (Eqs. 12, 13):

    s_{p+1/2}     = (s_{p+1} + s_p) / |s_{p+1} + s_p|
    s_dot_{p+1/2} = (delta_p / dt) (s_{p+1} - s_p) / |s_{p+1} - s_p|

with `delta_p` the angle between `s_p` and `s_{p+1}`, so the velocity magnitude is the
finite-difference angular velocity and its direction is orthogonal to the midpoint.

Minimization runs on the curved manifold (the product of unit spheres) with the tangent-projected
gradient (Eq. 15):

    grad_perp_p = grad_p - s_p (s_p . grad_p)

The gradient is a vectorized central finite difference made local: an interior image appears in only
its two neighbouring midpoint fields, so its gradient is evaluated from those two intervals alone,
which makes each iteration O(Q) rather than O(Q^2). The optimizer is a projected gradient descent with
a backtracking line search and geodesic retraction (each step is taken in the tangent plane and mapped
back to the sphere by normalization, the first-order retraction, which is what the reference's
velocity-projection optimizer does in spirit). Convergence is judged by the relative decrease in cost,
which is dimensionless, rather than by a unit-dependent gradient magnitude.

Validation
----------
On a uniaxial system this solver must reproduce the closed-form cost and pulse of
:mod:`spinoct.analytic.uniaxial` to tight tolerance. That is the acceptance gate in
``tests/test_image_ocp.py``; a solver that does not pass it is not trusted on biaxial or lattice
problems where no closed form exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dynamics.llg import field_from_trajectory
from ..dynamics.system import MacrospinSystem

__all__ = ["ImageOCPResult", "ImageOCPSolver"]


#: Resolution rule for :meth:`ImageOCPSolver.recommended_images`: the largest geodesic step per
#: interval (rad), the samples used to measure the path's arc, and the floor on the image count.
#: Measured at T = 100 tau0 with the L-BFGS minimizer: 0.15 rad leaves the cost 1.4 per cent above the
#: closed form, 0.1 rad leaves 0.7 per cent, and finer grids do not improve further (the iteration
#: budget binds, not the discretization).
_MAX_STEP_ANGLE = 0.1
_ARC_SAMPLES = 4001
_MIN_IMAGES = 60


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Project vectors onto the unit sphere, row-wise."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms


def _geodesic_interpolate(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    """A chain of `count` unit vectors along the great-circle arc from `start` to `end`.

    Args:
        start: unit vector, shape ``(3,)``.
        end: unit vector, shape ``(3,)``.
        count: number of images, at least 2, including both endpoints.

    Returns:
        Shape ``(count, 3)``, the endpoints reproduced exactly.
    """
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = np.arccos(dot)
    fractions = np.linspace(0.0, 1.0, count)
    if angle < 1e-9:
        return np.tile(start, (count, 1))
    if abs(angle - np.pi) < 1e-6:
        # Antipodal: the great circle is not unique. Pick an orthogonal axis and bend through it so
        # the chain does not collapse to a straight line through the origin (which would leave the
        # sphere). The reversal problems that matter here are exactly antipodal (north to south pole).
        reference = np.array([1.0, 0.0, 0.0]) if abs(start[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        orthogonal = reference - start * np.dot(reference, start)
        orthogonal /= np.linalg.norm(orthogonal)
        return np.stack(
            [np.cos(np.pi * f) * start + np.sin(np.pi * f) * orthogonal for f in fractions]
        )
    sin_angle = np.sin(angle)
    return np.stack(
        [
            (np.sin((1.0 - f) * angle) * start + np.sin(f * angle) * end) / sin_angle
            for f in fractions
        ]
    )


@dataclass(frozen=True)
class ImageOCPResult:
    """The outcome of a numerical optimal control path solve.

    Attributes:
        images: the converged chain, shape ``(Q + 2, 3)``, endpoints included.
        times: the time of each image, shape ``(Q + 2,)``.
        cost: the switching cost of the converged path, T^2 s.
        field_midpoints: the optimal field at each interval midpoint, shape ``(Q + 1, 3)``, T.
        converged: whether the descent reached the relative-cost tolerance (or a line-search minimum)
            before the iteration cap.
        iterations: how many descent steps were taken.
        final_force: the final maximum tangent-gradient magnitude, reported for diagnostics only; it
            carries units and is not the convergence criterion.
        history: the cost at each accepted iteration, for diagnostics.
    """

    images: np.ndarray
    times: np.ndarray
    cost: float
    field_midpoints: np.ndarray
    converged: bool
    iterations: int
    final_force: float
    history: np.ndarray

    def midpoint_times(self) -> np.ndarray:
        """The times at the interval midpoints, shape ``(Q + 1,)``."""
        return 0.5 * (self.times[1:] + self.times[:-1])


class ImageOCPSolver:
    """Solve the optimal control path by direct minimization of the discretized cost.

    Args:
        system: the magnetic system. Any :class:`MacrospinSystem`, uniaxial or biaxial.
        n_images: the number of interior movable images ``Q``. The reference uses up to 1500; a few
            hundred already resolve the macrospin path.
        switching_time: ``T`` in s.

    Notes:
        Determinism: the seed is drawn before any random perturbation of the initial chain, and the
        solver is otherwise deterministic. Use :meth:`solve_best` in practice, which sweeps several
        seeds and keeps the lowest cost, because multiple optimal control paths coexist in general and
        a single seed can report a local minimum (the biaxial case).
    """

    @staticmethod
    def recommended_images(system: MacrospinSystem, switching_time: float) -> int:
        """An image count that resolves the optimal path at this switching time.

        The discretization error of the midpoint rule is set by how far the moment moves between
        images, and the optimal path spirals: its geodesic arc grows with the switching time, so a fixed
        image count silently degrades as ``T`` rises. Measured on the uniaxial oracle at 60 images, the
        numerical cost sits 0.0 per cent above the closed form at ``T = 2 tau0`` and 20 per cent above it
        at ``T = 100 tau0``.

        The rule samples the closed-form path and asks for no more than ``_MAX_STEP_ANGLE`` radians per
        interval, with a floor of 60 images; with the L-BFGS minimizer that keeps the numerical cost
        within about one per cent of the closed form across the switching times the product bakes.
        For a biaxial system the uniaxial path of the same easy-axis
        anisotropy is used as the estimate, which is the right order because the arc is set by the
        precession the pulse has to follow.

        Args:
            system: the magnetic system.
            switching_time: ``T`` in s.

        Returns:
            The interior image count ``Q``.
        """
        from ..analytic.uniaxial import SwitchingTimeTooLongError, UniaxialOptimalControl

        uniaxial = MacrospinSystem(
            mu=system.mu, anisotropy_j=system.anisotropy_j, alpha=system.alpha, gamma=system.gamma
        )
        try:
            optimum = UniaxialOptimalControl.for_switching_time(uniaxial, switching_time)
        except SwitchingTimeTooLongError:
            return _MIN_IMAGES
        grid = np.linspace(0.0, switching_time, _ARC_SAMPLES)
        path = optimum.moment(grid)
        left, right = path[:-1], path[1:]
        arc = float(
            np.sum(
                2.0
                * np.arctan2(
                    np.linalg.norm(right - left, axis=-1), np.linalg.norm(right + left, axis=-1)
                )
            )
        )
        return int(max(_MIN_IMAGES, math.ceil(arc / _MAX_STEP_ANGLE)))

    def __init__(self, system: MacrospinSystem, n_images: int, switching_time: float) -> None:
        if n_images < 1:
            raise ValueError("n_images (Q) must be at least 1")
        if switching_time <= 0.0:
            raise ValueError("switching_time must be positive (s)")
        self.system = system
        self.n_images = n_images
        self.switching_time = switching_time
        self.times = np.linspace(0.0, switching_time, n_images + 2)
        self._dt = self.times[1] - self.times[0]

    # ------------------------------------------------------------------ the cost

    def _pair_field_squared(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """The squared optimal field ``|B|^2`` at the midpoint of each ``(left, right)`` image pair.

        Args:
            left: shape ``(N, 3)`` unit vectors.
            right: shape ``(N, 3)`` unit vectors.

        Returns:
            Shape ``(N,)``, the squared field in T^2, one per pair.
        """
        midpoints = _normalize(left + right)
        chord = right - left
        chord_norm = np.linalg.norm(chord, axis=-1, keepdims=True)
        # Geodesic angle, exact at every angle (arccos of the dot loses precision for close images).
        delta = 2.0 * np.arctan2(chord_norm[:, 0], np.linalg.norm(left + right, axis=-1))
        safe = chord_norm[:, 0] > 0.0
        direction = np.zeros_like(chord)
        direction[safe] = chord[safe] / chord_norm[safe]
        velocity = (delta / self._dt)[:, None] * direction
        fields = field_from_trajectory(midpoints, velocity, self.system)
        return np.sum(fields**2, axis=-1)

    def _midpoint_fields(self, images: np.ndarray) -> np.ndarray:
        """The optimal field vector at each interval midpoint, from the inverted equation of motion."""
        left = images[:-1]
        right = images[1:]
        midpoints = _normalize(left + right)
        chord = right - left
        chord_norm = np.linalg.norm(chord, axis=-1, keepdims=True)
        # Geodesic angle, exact at every angle (arccos of the dot loses precision for close images).
        delta = 2.0 * np.arctan2(chord_norm[:, 0], np.linalg.norm(left + right, axis=-1))
        safe = chord_norm[:, 0] > 0.0
        direction = np.zeros_like(chord)
        direction[safe] = chord[safe] / chord_norm[safe]
        velocity = (delta / self._dt)[:, None] * direction
        return field_from_trajectory(midpoints, velocity, self.system)

    def cost_of(self, images: np.ndarray) -> float:
        """The discretized switching cost of a chain, T^2 s."""
        left = images[:-1]
        right = images[1:]
        return float(np.sum(self._pair_field_squared(left, right)) * self._dt)

    # ------------------------------------------------------------------ the gradient

    def _tangent_gradient(self, chain: np.ndarray) -> np.ndarray:
        """The tangent-projected cost gradient with respect to the interior images.

        Local and vectorized. Interior image ``i`` appears in exactly two midpoint fields, the one
        with its left neighbour and the one with its right neighbour, so its gradient depends only on
        those two intervals. Holding every other image fixed, all ``Q`` interior images are perturbed
        in one component at once and their two-interval contributions are differenced in a batch. That
        makes each iteration ``O(Q)`` rather than the ``O(Q^2)`` of a naive full-cost finite
        difference, which is the difference between a solver that runs in seconds and one that does
        not (the reference uses up to 1500 images).
        """
        interior = chain[1:-1]
        left_neighbour = chain[:-2]
        right_neighbour = chain[2:]
        step = 1e-7  # dimensionless perturbation of a unit-vector component, re-projected each step

        def local_pair_cost(perturbed_interior: np.ndarray) -> np.ndarray:
            return self._pair_field_squared(left_neighbour, perturbed_interior) + self._pair_field_squared(
                perturbed_interior, right_neighbour
            )

        grad = np.zeros_like(interior)
        for component in range(3):
            forward = interior.copy()
            forward[:, component] += step
            backward = interior.copy()
            backward[:, component] -= step
            grad[:, component] = (local_pair_cost(forward) - local_pair_cost(backward)) / (2.0 * step)
        grad *= self._dt

        # Project each row onto the tangent plane of its image (Eq. 15).
        longitudinal = np.sum(grad * interior, axis=-1, keepdims=True)
        return grad - longitudinal * interior

    # ------------------------------------------------------------------ the minimizers

    def _minimize_lbfgs(
        self,
        start: np.ndarray,
        end: np.ndarray,
        interior: np.ndarray,
        max_iterations: int,
        relative_tolerance: float,
    ) -> tuple[np.ndarray, list[float], bool, int]:
        """L-BFGS over unconstrained vectors ``x`` with ``s = x / |x|``.

        The cost depends on ``x`` only through its direction, so the Euclidean gradient is the
        tangent-projected sphere gradient divided by ``|x|``. The objective is divided by the cost of the
        starting chain so the optimizer sees order-unity numbers rather than the 1e-12 T^2 s of a
        physical cost, which would otherwise meet its tolerances before taking a step.
        """
        from scipy.optimize import minimize

        shape = interior.shape
        scale = max(self.cost_of(np.vstack([start, interior, end])), np.finfo(float).tiny)
        history: list[float] = []

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            raw = vector.reshape(shape)
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            chain = np.vstack([start, raw / norms, end])
            cost = self.cost_of(chain)
            history.append(cost)
            grad = self._tangent_gradient(chain) / norms
            return cost / scale, grad.reshape(-1) / scale

        result = minimize(
            objective,
            interior.reshape(-1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "ftol": relative_tolerance, "gtol": 1e-14},
        )
        return _normalize(result.x.reshape(shape)), history, bool(result.success), int(result.nit)

    # ------------------------------------------------------------------ the solve

    def solve(
        self,
        max_iterations: int = 4000,
        relative_tolerance: float = 1e-12,
        seed: int | None = 0,
        noise: float = 0.05,
        method: str = "lbfgs",
    ) -> ImageOCPResult:
        """Minimize the cost over the interior images.

        Args:
            method: ``"lbfgs"`` (default), a quasi-Newton minimization over unconstrained vectors with
                ``s = x / |x|``, or ``"descent"``, the projected gradient descent with geodesic
                retraction kept as the reference. Descent converges slowly on long switching times,
                where the path spirals: at ``T = 100 tau0`` it sat 12 per cent above the closed form
                after 2500 iterations and was still 4 per cent above it after 40000.
            max_iterations: the descent iteration cap.
            relative_tolerance: convergence when the relative decrease in cost between successive
                accepted steps falls below this. **Dimensionless on purpose.** A gradient-magnitude
                tolerance would be unit-dependent, since the gradient carries units of cost per
                radian, and the natural scale of the cost varies over orders of magnitude with the
                switching time; a relative cost criterion is the same everywhere.
            seed: the random seed for the initial symmetry-breaking perturbation. Drawn before the
                chain is built, so the result is reproducible. Sweep several seeds when multiple
                optimal control paths may coexist (the biaxial case) and keep the lowest cost.
            noise: the standard deviation, in radians of arc, of the perturbation added to the initial
                great-circle chain. The default is nonzero and deliberately so: the pole-to-pole
                geodesic lies in a symmetry plane where the gradient vanishes to first order, so a
                solve started exactly on it reports the meridian as converged even though the true
                optimal control path precesses away from it. Breaking the symmetry is not optional,
                it is a correctness requirement the reference states explicitly.

        Returns:
            The :class:`ImageOCPResult`.
        """
        rng = np.random.default_rng(seed)
        start = self.system_start()
        end = self.system_end()
        chain = _geodesic_interpolate(start, end, self.n_images + 2)

        if noise > 0.0:
            perturbation = rng.normal(scale=noise, size=(self.n_images, 3))
            chain[1:-1] = _normalize(chain[1:-1] + perturbation)

        interior = chain[1:-1].copy()

        if method == "lbfgs":
            interior, history, converged, iterations_done = self._minimize_lbfgs(
                start, end, interior, max_iterations, relative_tolerance
            )
            chain = np.vstack([start, interior, end])
            return ImageOCPResult(
                images=chain,
                times=self.times,
                cost=self.cost_of(chain),
                field_midpoints=self._midpoint_fields(chain),
                converged=converged,
                iterations=iterations_done,
                final_force=float(np.max(np.linalg.norm(self._tangent_gradient(chain), axis=-1))),
                history=np.asarray(history),
            )
        if method != "descent":
            raise ValueError("method must be 'lbfgs' or 'descent'")

        # An adaptive step size with backtracking. The cost surface is smooth but the natural scale of
        # the gradient varies over orders of magnitude with the switching time, so a fixed step is
        # brittle; backtracking makes the solver robust without a hand-tuned learning rate.
        step_size = 1.0
        history = [self.cost_of(chain)]
        converged = False
        iterations_done = 0

        for iteration in range(1, max_iterations + 1):
            iterations_done = iteration
            grad = self._tangent_gradient(np.vstack([start, interior, end]))
            current_cost = history[-1]

            # Backtracking line search along the negative projected gradient, with geodesic
            # retraction (normalize back to the sphere after the tangent step).
            trial_step = step_size
            improved = False
            for _ in range(40):
                trial_interior = _normalize(interior - trial_step * grad)
                trial_cost = self.cost_of(np.vstack([start, trial_interior, end]))
                if trial_cost < current_cost:
                    improved = True
                    break
                trial_step *= 0.5

            if not improved:
                # No decrease found in any direction the step can reach: at a minimum to resolution.
                converged = True
                break

            interior = trial_interior
            history.append(trial_cost)
            if (current_cost - trial_cost) <= relative_tolerance * current_cost:
                converged = True
                break
            # Grow the step gently when progress is easy, so long shallow descents do not crawl.
            step_size = trial_step * 1.2

        chain = np.vstack([start, interior, end])
        fields = self._midpoint_fields(chain)
        cost = self.cost_of(chain)
        final_force = float(np.max(np.linalg.norm(self._tangent_gradient(chain), axis=-1)))
        return ImageOCPResult(
            images=chain,
            times=self.times,
            cost=cost,
            field_midpoints=fields,
            converged=converged,
            iterations=iterations_done,
            final_force=final_force,
            history=np.asarray(history),
        )

    def solve_best(
        self,
        n_seeds: int = 8,
        base_seed: int = 0,
        noise: float = 0.15,
        **solve_kwargs: object,
    ) -> ImageOCPResult:
        """Run several seeded solves and return the lowest-cost optimal control path.

        This is the interface to use in practice, not :meth:`solve`. Multiple optimal control paths
        coexist in general, and for a biaxial system the asymmetric ones can be the global optimum
        while the symmetric one is only a local minimum (Phys. Rev. B 107, 214448, which reports up
        to six coexisting paths for one parameter set). A single seed therefore reports whichever
        basin it happened to fall into, which may not be the cheapest. Sweeping seeds and keeping the
        best is the reference's own prescription and this package's guard against silently reporting a
        local minimum as the answer.

        Args:
            n_seeds: how many independent seeds to try.
            base_seed: the first seed; the sweep uses ``base_seed`` through ``base_seed + n_seeds - 1``,
                so the whole sweep is reproducible.
            noise: the symmetry-breaking perturbation scale for the sweep, in radians. Larger than the
                single-solve default because the point of the sweep is to explore distinct basins.
            **solve_kwargs: forwarded to :meth:`solve` (for example ``max_iterations``).

        Returns:
            The lowest-cost :class:`ImageOCPResult` over the sweep.
        """
        if n_seeds < 1:
            raise ValueError("n_seeds must be at least 1")
        best: ImageOCPResult | None = None
        for offset in range(n_seeds):
            candidate = self.solve(seed=base_seed + offset, noise=noise, **solve_kwargs)
            if best is None or candidate.cost < best.cost:
                best = candidate
        assert best is not None
        return best

    # ------------------------------------------------------------------ endpoints

    def system_start(self) -> np.ndarray:
        """The initial state, the easy-axis north pole ``(0, 0, +1)``."""
        return np.array([0.0, 0.0, 1.0])

    def system_end(self) -> np.ndarray:
        """The final state, the easy-axis south pole ``(0, 0, -1)``."""
        return np.array([0.0, 0.0, -1.0])
