"""The free optimal control path of a spin chain: direct minimization over every site's trajectory.

The two-mode comparison in :mod:`spinoct.lattice.reversal` asks which of two fixed reversal modes is
cheaper. This module asks the real question the macrospin literature leaves open (Phys. Rev. B 107,
214448, 2023): with every site free to follow its own trajectory, what reversal does the optimal
control actually choose, and is it ever cheaper than rotating the whole chain uniformly?

Method
------
The image-based direct minimization of Badarneh, Kwiatkowski and Bessarab (Phys. Rev. B 107, 214448)
generalized from one moment to ``N`` coupled moments. The trajectory is an array of images of shape
``(Q + 2, N, 3)``: ``Q + 2`` time slices, ``N`` sites, unit vectors. The first slice is all up, the
last is all down, both clamped. At the midpoint of every interval and every site the midpoint position
and velocity follow the single-site midpoint rule, and the applied field that site needs comes from
inverting the equation of motion with the CHAIN internal field (anisotropy plus exchange from the
neighbours, evaluated at the midpoint configuration):

    b_i = (alpha / gamma) v_i + (1 / gamma) (s_i x v_i) - b_int,i_perp.

The cost is the summed squared field over sites and intervals, times the interval length.

Gradient by graph coloring
--------------------------
An interior image ``(q, i)`` affects only two intervals in time (``q - 1`` and ``q``) and, through the
exchange field, three sites in space (``i - 1``, ``i``, ``i + 1``). So images two apart in time and three
apart in space have disjoint footprints. Perturbing every image of one color class at once, and reading
each image's cost change off its own footprint, gives the exact central-difference gradient for all
images in ``2 x 3 = 6`` color classes, times three components, times two signs: 36 full cost
evaluations per gradient, independent of ``Q`` and ``N``. Each cost evaluation is itself fully
vectorized.

Validation
----------
A one-site chain with no exchange must reproduce the macrospin image-based solver and the analytic
cost. And because uniform rotation is a feasible trajectory that never pays exchange (neighbours stay
parallel, so the exchange field is longitudinal and drops out of the transverse projection), the free
optimum can never cost more than ``N`` times the macrospin optimum. Both are enforced in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import SpinChain

__all__ = ["LatticeOCPResult", "LatticeOCPSolver"]

#: Central-difference step for perturbing a unit-vector component. Dimensionless; the perturbed image
#: is re-projected onto the sphere by the retraction.
_FD_STEP = 1e-7

#: Temporal colouring period: an image touches 2 intervals. The spatial colouring comes from the
#: lattice (3 classes on a chain, 5 on a square patch).
_TIME_COLORS = 2

#: Dense samples of the analytic single-site optimum used to build seeds by interpolation.
_SEED_SAMPLES = 4001

#: Iteration cap and relative cost tolerance for the single-site reference solve behind the bound.
_BOUND_MAX_ITERATIONS = 5000
_BOUND_TOLERANCE = 1e-13

#: Images along the minimum energy path used to build the ``"mep"`` seed.
_MEP_IMAGES = 65

#: Resolution rule for :meth:`LatticeOCPSolver.recommended_images`: largest geodesic step of the
#: single-site optimum per interval (rad), largest wall displacement per interval (in wall widths),
#: and the floor on the image count.
_MAX_STEP_ANGLE = 0.15
_MAX_WALL_STEP = 0.25
_MIN_IMAGES = 40


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@dataclass(frozen=True)
class LatticeOCPResult:
    """The outcome of a free chain optimal-control solve.

    Attributes:
        images: the converged trajectory, shape ``(Q + 2, N, 3)``.
        times: the slice times, s, shape ``(Q + 2,)``.
        cost: the switching cost, T^2 s.
        uniform_bound: ``N`` times the macrospin optimal cost for the same discretization, T^2 s. The
            free optimum can never exceed it.
        nonuniformity: the time-averaged spread of the polar angle across sites, radians. Zero for a
            perfectly uniform rotation; large for a wall or a spin wave.
        converged: whether the relative-cost criterion was met.
        iterations: descent steps taken.
    """

    images: np.ndarray
    times: np.ndarray
    cost: float
    uniform_bound: float
    nonuniformity: float
    converged: bool
    iterations: int

    @property
    def saving_vs_uniform(self) -> float:
        """The fractional saving over uniform rotation, ``1 - cost / uniform_bound``."""
        return 1.0 - self.cost / self.uniform_bound if self.uniform_bound > 0 else 0.0


class LatticeOCPSolver:
    """Minimize the switching cost over all trajectories of a spin chain.

    Args:
        chain: the spin chain.
        n_images: interior time slices ``Q``.
        switching_time: ``T`` in s.
    """

    def __init__(self, chain: SpinChain, n_images: int, switching_time: float) -> None:
        if n_images < 1:
            raise ValueError("n_images must be at least 1")
        if switching_time <= 0.0:
            raise ValueError("switching_time must be positive (s)")
        self.chain = chain
        self.n_images = n_images
        self.switching_time = switching_time
        self.times = np.linspace(0.0, switching_time, n_images + 2)
        self._dt = self.times[1] - self.times[0]
        self._uniform_bound: float | None = None
        self._uniform_images: np.ndarray | None = None

    # ------------------------------------------------------------------ the field and the cost

    def _internal_field_batched(self, spins: np.ndarray) -> np.ndarray:
        """The lattice's internal field for a stack of configurations, shape ``(P, N, 3)``."""
        return self.chain.internal_field_batched(spins)

    def cost_matrix(self, images: np.ndarray) -> np.ndarray:
        """The squared applied field per interval and site, times ``dt``, shape ``(Q + 1, N)``."""
        left = images[:-1]
        right = images[1:]
        mid = _normalize(left + right)
        chord = right - left
        chord_norm = np.linalg.norm(chord, axis=-1, keepdims=True)
        # The geodesic angle as 2 atan2(|r - l|, |r + l|), exact at every angle. arccos of the dot
        # product loses all precision when neighbouring images nearly coincide (a site parked at a
        # pole), which turned the finite-difference gradient there into rounding noise and stalled
        # wall-seeded solves far above the optimum.
        delta = 2.0 * np.arctan2(chord_norm[..., 0], np.linalg.norm(left + right, axis=-1))
        direction = np.divide(chord, chord_norm, out=np.zeros_like(chord), where=chord_norm > 0.0)
        velocity = (delta / self._dt)[..., None] * direction

        internal = self._internal_field_batched(mid)
        internal_perp = internal - np.sum(internal * mid, axis=-1, keepdims=True) * mid
        alpha, gamma = self.chain.alpha, self.chain.gamma
        field = (alpha / gamma) * velocity + (1.0 / gamma) * np.cross(mid, velocity) - internal_perp
        return np.sum(field**2, axis=-1) * self._dt

    def cost_of(self, images: np.ndarray) -> float:
        """The total switching cost, T^2 s."""
        return float(np.sum(self.cost_matrix(images)))

    # ------------------------------------------------------------------ the colored gradient

    def _gradient(self, images: np.ndarray) -> np.ndarray:
        """Exact central-difference gradient for every interior image, tangent-projected."""
        q_total, n_sites = images.shape[0], images.shape[1]
        grad = np.zeros_like(images)
        interior_q = np.arange(1, q_total - 1)
        sites = np.arange(n_sites)
        # The lattice supplies a colouring whose classes have disjoint closed neighbourhoods: three
        # classes on a chain, five on a square patch.
        colors = self.chain.colors()

        for tc in range(_TIME_COLORS):
            q_sel = interior_q[(interior_q % _TIME_COLORS) == tc]
            if q_sel.size == 0:
                continue
            for sc in range(int(colors.max()) + 1):
                i_sel = sites[colors == sc]
                if i_sel.size == 0:
                    continue
                qq, ii = np.meshgrid(q_sel, i_sel, indexing="ij")
                for component in range(3):
                    plus = images.copy()
                    plus[qq, ii, component] += _FD_STEP
                    minus = images.copy()
                    minus[qq, ii, component] -= _FD_STEP
                    diff = self.cost_matrix(plus) - self.cost_matrix(minus)  # (Q+1, N)
                    # Sum each image's footprint: intervals q-1, q and the site's closed neighbourhood.
                    spatial = self.chain.footprint_sum(diff)  # (Q+1, N)
                    footprint = spatial[qq - 1, ii] + spatial[qq, ii]
                    grad[qq, ii, component] = footprint / (2.0 * _FD_STEP)

        tangential = grad - np.sum(grad * images, axis=-1, keepdims=True) * images
        tangential[0] = 0.0
        tangential[-1] = 0.0
        return tangential

    # ------------------------------------------------------------------ initial paths

    def _macrospin_moment(self, times: np.ndarray) -> np.ndarray:
        """The single-site optimal path sampled at ``times``, shape ``(len(times), 3)``.

        The closed-form uniaxial optimum (PRL 126, 177206) when the switching time is representable;
        otherwise the pole-to-pole great-circle arc through +x. The analytic path carries the correct
        azimuthal precession, so a solve seeded from it starts in the basin of the true optimum rather
        than in a neighbouring local minimum with a different precession history.
        """
        from ..analytic.uniaxial import SwitchingTimeTooLongError, UniaxialOptimalControl
        from ..dynamics.system import MacrospinSystem

        macrospin = MacrospinSystem(
            mu=self.chain.mu,
            anisotropy_j=self.chain.anisotropy_j,
            alpha=self.chain.alpha,
            gamma=self.chain.gamma,
        )
        try:
            optimum = UniaxialOptimalControl.for_switching_time(macrospin, self.switching_time)
        except SwitchingTimeTooLongError:
            fractions = np.clip(times / self.switching_time, 0.0, 1.0)
            return np.stack(
                [np.sin(np.pi * fractions), np.zeros_like(fractions), np.cos(np.pi * fractions)], -1
            )
        dense = np.linspace(0.0, self.switching_time, _SEED_SAMPLES)
        path = optimum.moment(dense)
        index = np.clip(times / self.switching_time, 0.0, 1.0) * (_SEED_SAMPLES - 1)
        return _normalize(
            np.stack([np.interp(index, np.arange(_SEED_SAMPLES), path[:, k]) for k in range(3)], -1)
        )

    def _uniform_path(self) -> np.ndarray:
        """Every site on the same single-site optimal path: a feasible, exchange-free trajectory."""
        path = self._macrospin_moment(self.times)
        path[0] = np.array([0.0, 0.0, 1.0])
        path[-1] = np.array([0.0, 0.0, -1.0])
        return np.repeat(path[:, None, :], self.chain.n_sites, axis=1)

    @staticmethod
    def recommended_images(chain: SpinChain, switching_time: float) -> int:
        """An interior image count that resolves both the precession and a wall crossing the chain.

        The single-site optimum's total geodesic arc is sampled at no more than ``0.15`` rad per
        interval (coarser grids alias the precession and produce spurious minima far above the bound),
        and a wall of the chain's width crossing ``N + 4 w`` sites moves at most a quarter width per
        interval. Never fewer than 40.

        Args:
            chain: the spin chain.
            switching_time: ``T`` in s.

        Returns:
            The interior image count ``Q``.
        """
        probe = LatticeOCPSolver(chain, 1, switching_time)
        path = probe._macrospin_moment(np.linspace(0.0, switching_time, _SEED_SAMPLES))
        arc = float(np.sum(2.0 * np.arctan2(
            np.linalg.norm(path[1:] - path[:-1], axis=-1), np.linalg.norm(path[1:] + path[:-1], axis=-1)
        )))
        width = max(1.0, probe.wall_width_sites())
        by_arc = np.ceil(arc / _MAX_STEP_ANGLE)
        by_wall = np.ceil((chain.wall_extent() + 4.0 * width) / (_MAX_WALL_STEP * width))
        return int(max(_MIN_IMAGES, by_arc, by_wall))

    def wall_width_sites(self) -> float:
        """The continuum domain-wall width ``sqrt(J / (2 K))``, in lattice sites.

        From the energy density ``(J/2)(d theta/dx)^2 + K sin^2 theta`` of the chain Hamiltonian.
        """
        return float(np.sqrt(self.chain.exchange_j / (2.0 * self.chain.anisotropy_j)))

    def _wall_path(self) -> np.ndarray:
        """A domain wall that enters at site 0 and leaves past site N-1 over the window.

        The polar angle of each site is a tanh profile of the wall centre, ``theta_i = pi * progress_i``;
        the azimuth of every site is the azimuth of the single-site optimum at the same instant, so the
        whole chain precesses together and only the reversal is staggered in space.

        Staggering the full optimal path in time instead (site ``i`` at time ``t`` sitting where the
        optimum is at its own delayed progress) is a trap: it compresses the optimum's precession into
        each site's short flip window, so a site jumps almost antipodally between consecutive images,
        where the midpoint rule is singular. That seed started at 12000 times the uniform bound and
        pinned the optimizer at 51 times it.
        """
        extent = self.chain.wall_extent()
        fractions = np.linspace(0.0, 1.0, self.n_images + 2)
        sites = self.chain.wall_coordinate()
        width = max(1.0, self.wall_width_sites())
        centre = -2.0 * width + (extent - 1 + 4.0 * width) * fractions
        progress = 0.5 * (1.0 + np.tanh((centre[:, None] - sites[None, :]) / width))
        progress = (progress - progress[0]) / (progress[-1] - progress[0])
        theta = np.pi * progress
        reference = self._macrospin_moment(self.times)
        phi = np.arctan2(reference[:, 1], reference[:, 0])[:, None]
        path = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )
        path[0] = np.array([0.0, 0.0, 1.0])
        path[-1] = np.array([0.0, 0.0, -1.0])
        return path

    def _mep_path(self) -> np.ndarray:
        """The minimum energy path geometry, traversed at the pace of the single-site optimum.

        The chain follows the wall MEP of :func:`spinoct.lattice.minimum_energy_path` (which lies in
        the x-z plane), its progress along the path at each instant equal to the fraction of the polar
        reversal the single-site optimum has completed, and the whole chain rotated about z by the
        optimum's azimuth. For long chains at long switching times this seed reaches the nonuniform
        optimum where the uniform and tanh-wall seeds stall.
        """
        from .mep import minimum_energy_path

        path = minimum_energy_path(self.chain, n_images=_MEP_IMAGES, initial="wall").images
        reference = self._macrospin_moment(self.times)
        schedule = np.arccos(np.clip(reference[:, 2], -1.0, 1.0)) / np.pi
        index = schedule * (path.shape[0] - 1)
        lower = np.clip(np.floor(index).astype(int), 0, path.shape[0] - 2)
        weight = (index - lower)[:, None, None]
        geometry = _normalize((1.0 - weight) * path[lower] + weight * path[lower + 1])
        phi = np.arctan2(reference[:, 1], reference[:, 0])
        cos_phi, sin_phi = np.cos(phi)[:, None], np.sin(phi)[:, None]
        out = np.empty_like(geometry)
        out[..., 0] = cos_phi * geometry[..., 0] - sin_phi * geometry[..., 1]
        out[..., 1] = sin_phi * geometry[..., 0] + cos_phi * geometry[..., 1]
        out[..., 2] = geometry[..., 2]
        out[0] = np.array([0.0, 0.0, 1.0])
        out[-1] = np.array([0.0, 0.0, -1.0])
        return out

    # ------------------------------------------------------------------ the solve

    def solve(
        self,
        initial: str = "uniform",
        seed: int = 0,
        noise: float = 0.05,
        max_iterations: int = 3000,
        relative_tolerance: float = 1e-10,
        method: str = "lbfgs",
    ) -> LatticeOCPResult:
        """Minimize the chain switching cost over every site's trajectory.

        Args:
            initial: ``"uniform"`` (the single-site optimum on every site), ``"wall"`` (a tanh wall on the
                optimum's azimuth) or ``"mep"`` (the minimum energy path on the optimum's schedule).
            method: ``"lbfgs"`` (default; quasi-Newton on normalized vectors) or ``"descent"``
                (projected gradient descent with backtracking, kept as the reference).
            seed: seed for the symmetry-breaking perturbation, drawn before the path is built.
            noise: perturbation scale in radians. Nonzero by default, because the pole-to-pole arc is a
                saddle, as in the macrospin solver.
            max_iterations: iteration cap.
            relative_tolerance: stop when the relative cost decrease falls below this (dimensionless).

        Returns:
            The :class:`LatticeOCPResult`.
        """
        rng = np.random.default_rng(seed)
        if initial == "uniform":
            images = self.uniform_images()
        elif initial == "wall":
            images = self._wall_path()
        elif initial == "mep":
            images = self._mep_path()
        else:
            raise ValueError("initial must be 'uniform', 'wall' or 'mep'")
        if noise > 0.0:
            perturbation = rng.normal(scale=noise, size=images[1:-1].shape)
            images[1:-1] = _normalize(images[1:-1] + perturbation)

        if method == "lbfgs":
            images, cost, converged, iterations = self._minimize_lbfgs(
                images, max_iterations, relative_tolerance
            )
        elif method == "descent":
            images, cost, converged, iterations = self._minimize_descent(
                images, max_iterations, relative_tolerance
            )
        else:
            raise ValueError("method must be 'lbfgs' or 'descent'")

        theta = np.arccos(np.clip(images[..., 2], -1.0, 1.0))
        nonuniformity = float(np.mean(np.std(theta[1:-1], axis=1)))
        return LatticeOCPResult(
            images=images,
            times=self.times,
            cost=cost,
            uniform_bound=self.uniform_bound(),
            nonuniformity=nonuniformity,
            converged=converged,
            iterations=iterations,
        )

    def _minimize_lbfgs(
        self, images: np.ndarray, max_iterations: int, relative_tolerance: float
    ) -> tuple[np.ndarray, float, bool, int]:
        """L-BFGS over unconstrained interior vectors ``x`` with ``s = x / |x|``.

        The cost depends on ``x`` only through its direction, so the Euclidean gradient in ``x`` is the
        tangent-projected sphere gradient divided by ``|x|``. The objective is divided by the cost of the
        analytic uniform path, so the optimizer sees O(1) numbers near the optimum instead of the order
        1e-12 T^2 s of a physical cost. Dividing by the SEED's cost instead is a trap: SciPy's relative
        cost test uses ``max(|f|, 1)``, so an expensive seed makes the optimum look like ``f << 1`` and
        the test passes long before convergence (observed: a wall seed stopping at 85 times the bound).
        """
        from scipy.optimize import minimize

        shape = images[1:-1].shape
        scale = max(self.cost_of(self._uniform_path()), np.finfo(float).tiny)
        work = images.copy()

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            raw = vector.reshape(shape)
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            work[1:-1] = raw / norms
            grad_sphere = self._gradient(work)[1:-1]
            return self.cost_of(work) / scale, (grad_sphere / norms).reshape(-1) / scale

        result = minimize(
            objective,
            images[1:-1].reshape(-1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "ftol": relative_tolerance, "gtol": 1e-12},
        )
        final = images.copy()
        final[1:-1] = _normalize(result.x.reshape(shape))
        return final, self.cost_of(final), bool(result.success), int(result.nit)

    def _minimize_descent(
        self, images: np.ndarray, max_iterations: int, relative_tolerance: float
    ) -> tuple[np.ndarray, float, bool, int]:
        """Projected gradient descent with a normalization retraction and backtracking (the reference)."""
        cost = self.cost_of(images)
        step = 1.0
        converged = False
        iterations = 0
        for _ in range(max_iterations):
            iterations += 1
            grad = self._gradient(images)
            # Normalize by the largest per-image gradient so the step is the largest move in radians.
            gnorm = float(np.max(np.linalg.norm(grad, axis=-1)))
            if gnorm == 0.0:
                converged = True
                break
            trial_step = step
            improved = False
            for _ in range(40):
                trial = images.copy()
                trial[1:-1] = _normalize(images[1:-1] - trial_step * grad[1:-1] / gnorm)
                trial_cost = self.cost_of(trial)
                if trial_cost < cost:
                    improved = True
                    break
                trial_step *= 0.5
            if not improved:
                converged = True
                break
            decrease = cost - trial_cost
            images, cost = trial, trial_cost
            if decrease <= relative_tolerance * cost:
                converged = True
                break
            step = min(trial_step * 1.5, 0.5)
        return images, cost, converged, iterations

    def uniform_images(self) -> np.ndarray:
        """The discrete single-site optimum on this grid, repeated on every site, shape ``(Q + 2, N, 3)``.

        One site with no exchange is solved on the same time grid and the same midpoint cost as the
        chain, seeded from the analytic optimum. Repeating it on every site is a feasible chain
        trajectory whose exchange field is purely longitudinal, so its chain cost is exactly ``N`` times
        the single-site cost. Cached.
        """
        if self._uniform_images is None:
            single_chain = SpinChain(
                n_sites=1,
                mu=self.chain.mu,
                anisotropy_j=self.chain.anisotropy_j,
                exchange_j=0.0,
                alpha=self.chain.alpha,
                gamma=self.chain.gamma,
            )
            single = LatticeOCPSolver(single_chain, self.n_images, self.switching_time)
            seed = single._uniform_path()
            images, _cost, _converged, _iterations = single._minimize_lbfgs(
                seed, _BOUND_MAX_ITERATIONS, _BOUND_TOLERANCE
            )
            self._uniform_images = np.repeat(images, self.chain.n_sites, axis=1)
        return self._uniform_images.copy()

    def uniform_bound(self) -> float:
        """The chain cost of :meth:`uniform_images`: an upper bound on the free optimum, T^2 s."""
        if self._uniform_bound is None:
            self._uniform_bound = self.cost_of(self.uniform_images())
        return self._uniform_bound

    def solve_best(self, n_seeds: int = 3, **kwargs: object) -> LatticeOCPResult:
        """Solve from uniform, wall and MEP starts over several seeds and keep the cheapest."""
        best: LatticeOCPResult | None = None
        for initial in ("uniform", "wall", "mep"):
            for seed in range(n_seeds):
                result = self.solve(initial=initial, seed=seed, **kwargs)
                if best is None or result.cost < best.cost:
                    best = result
        assert best is not None
        return best
