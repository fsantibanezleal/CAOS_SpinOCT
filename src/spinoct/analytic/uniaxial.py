"""The exact optimal control path for a uniaxial macrospin, and its closed-form pulse and cost.

Source
------
Kwiatkowski, G. J., Badarneh, M. H. A., Berkov, D. V., Bessarab, P. F.,
*Optimal Control of Magnetization Reversal in a Monodomain Particle by Means of Applied Magnetic
Field*, Phys. Rev. Lett. 126, 177206 (2021), https://doi.org/10.1103/PhysRevLett.126.177206.
Preprint arXiv:2004.02146.

The problem
-----------
Minimize the switching cost

    Phi = int_0^T |b(t)|^2 dt

over trajectories that carry the moment from ``s_z = +1`` to ``s_z = -1`` in time ``T``, subject to
the Landau-Lifshitz-Gilbert equation with ``E = -K s_z^2``. Phi is the Joule heating of the circuit
generating the field, not the energy the magnet dissipates; see
:mod:`spinoct.units` and ``docs/theory/03-what-the-cost-functional-measures.md``.

The solution
------------
Substituting the inverted equation of motion turns the constrained problem into an unconstrained
one over the trajectory, whose Euler-Lagrange equations separate in spherical coordinates:

    tau0^2 theta'' = alpha^2 / (4 (1 + alpha^2)^2) * sin(4 theta)
    tau0 phi'       = cos(theta) / (1 + alpha^2)

The first is the sine-Gordon equation, solved by the Jacobi amplitude

    theta(t) = (1/2) am( t / [p tau0 (1 + alpha^2)] | -alpha^2 p^2 )

with ``p`` fixed implicitly by the switching time through

    T = 4 tau0 (1 + alpha^2) p K(-alpha^2 p^2)

The optimal field is always perpendicular to the moment, with amplitude

    b(t) = K / (mu p sqrt(1 + alpha^2)) * [ dn(u | -alpha^2 p^2) + alpha p sn(u | -alpha^2 p^2) ]

where ``u = t / [p tau0 (1 + alpha^2)]``, and direction

    b_vec = b / sqrt(1 + alpha^2) * ( alpha e_theta + e_phi )

Four independent identities follow in closed form and are used as the package's positive controls.
Each is implemented here and asserted in ``tests/test_uniaxial_analytic.py``:

1. Pulse symmetry: ``b(0) = b(T/2) = b(T)``, because ``sn`` vanishes and ``dn`` is one at
   ``u = 0, 2K, 4K``.
2. Extremal spread: ``b_max - b_min = 2 alpha K / (mu sqrt(1 + alpha^2))``, with the maximum
   exactly at ``t = T/4`` and the minimum exactly at ``t = 3T/4``. This follows from
   ``f' = alpha p cn(u) f``, so the extrema sit where ``cn`` vanishes.
3. Mean amplitude: ``b_av = pi sqrt(1 + alpha^2) / (gamma T)``, independent of the magnetic
   potential, because ``int_0^{4K} dn du = 2 pi`` and ``int_0^{4K} sn du = 0``.
4. Minimum cost: ``Phi_m = 2 K [2 E(-a^2p^2) - K(-a^2p^2)] / (gamma mu p)``, because
   ``int_0^{4K} f^2 du = 8 E - 4 K``.

Identity 4 is exactly the published cost formula, derived here from the pulse rather than quoted,
which is what makes it a control rather than a restatement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..dynamics.system import MacrospinSystem
from .elliptic import complete_e, complete_k, jacobi_am, jacobi_snd

#: Iteration caps. Integers, so they carry no unit assumption; they exist only to turn a pathological
#: input into a clear error instead of a hang.
_MAX_BRACKET_DOUBLINGS = 200
_MAX_ROOT_ITERATIONS = 200

#: The published tolerance for the "optimal switching time" knee: the switching time at which the
#: cost sits this fraction above the infinite-time floor. Dimensionless, and 0.1 is the value used
#: in Phys. Rev. Lett. 126, 177206 (2021) Fig. 3.
DEFAULT_KNEE_TOLERANCE = 0.1

#: Relative tolerance for deciding whether a query grid starts at the initial time. Dimensionless:
#: it is compared against t/T, a ratio of two times, so it carries no unit of its own. It only
#: distinguishes "the grid begins at the origin" from "it begins somewhere inside the window".
_ORIGIN_RELATIVE_TOLERANCE = 1e-12

__all__ = [
    "DEFAULT_KNEE_TOLERANCE",
    "SwitchingTimeTooLongError",
    "UniaxialOptimalControl",
    "cost_free_macrospin",
    "cost_infinite_time",
    "optimal_switching_time",
    "solve_shape_parameter",
]


class SwitchingTimeTooLongError(ValueError):
    """The switching time is so long that the shape parameter is not representable in double precision.

    This is not a numerical accident to be worked around, it is a statement about the physics: the
    switching time grows only logarithmically in the shape parameter ``p``, so a very long ``T``
    demands an astronomically large ``p`` and the cost has already saturated onto its infinite-time
    floor to within machine epsilon. Callers who want the floor should read
    :func:`cost_infinite_time` directly; the optimal control path itself carries no further
    information there.
    """


def solve_shape_parameter(switching_time: float, system: MacrospinSystem) -> float:
    """Solve ``T = 4 tau0 (1 + alpha^2) p K(-alpha^2 p^2)`` for the shape parameter ``p``.

    Args:
        switching_time: the switching time ``T`` in s. Must be positive.
        system: the macrospin. Only ``alpha`` and ``tau0`` enter.

    Returns:
        The dimensionless shape parameter ``p``.

    Raises:
        ValueError: if the switching time is not positive.

    Notes:
        At zero damping the relation collapses to ``T = 2 pi tau0 p`` and is solved in closed form.

        For finite damping the right-hand side is strictly increasing in ``p``, so a bracketing root
        find is safe and the root is unique. The bracket is **derived**, not guessed: ``K(m) < K(0)
        = pi/2`` for every ``m < 0``, so ``T < 2 pi tau0 (1 + alpha^2) p`` and the zero-damping
        solution is a rigorous lower bound on ``p``. The upper end is found by doubling from there,
        which terminates because ``p K(-alpha^2 p^2)`` grows without bound (like ``ln(p)/alpha``).

        ``p`` spans many orders of magnitude across the switching times of interest, which is
        exactly why a fixed numerical bracket would have been a latent unit assumption rather than
        a convenience.
    """
    if switching_time <= 0.0:
        raise ValueError("switching_time must be positive (s)")

    alpha = system.alpha
    tau0 = system.tau0
    scale = 4.0 * tau0 * (1.0 + alpha**2)

    if alpha == 0.0:
        # K(0) = pi/2 exactly, so T = 2 pi tau0 p.
        return switching_time / (2.0 * math.pi * tau0)

    # The shape parameter is not representable for arbitrarily long switching times, and the reason
    # is physical rather than numerical. As p grows, K(-alpha^2 p^2) falls like ln(4 alpha p)/(alpha p),
    # so the period relation reads T ~ 4 tau0 (1 + alpha^2) ln(4 alpha p) / alpha: the switching time
    # grows only LOGARITHMICALLY in p. Asking for a long T therefore demands an astronomically large p.
    #
    # That is exactly the regime where the cost has already converged onto its floor: the published
    # asymptote is Phi_m = Phi_inf (1 + 4 exp[-alpha T / (2 tau0 (1 + alpha^2))]). Once that excess
    # falls below machine epsilon there is no information left in p, and returning a saturated root
    # would be reporting a number the arithmetic does not have. Refuse instead, and say why.
    excess_over_floor = 4.0 * math.exp(
        -alpha * switching_time / (2.0 * tau0 * (1.0 + alpha**2))
    )
    if excess_over_floor < float(np.finfo(float).eps):
        raise SwitchingTimeTooLongError(
            f"at T = {switching_time / tau0:.4g} tau0 with alpha = {alpha:g} the cost sits within "
            f"{excess_over_floor:.3g} of its infinite-time floor, which is below double precision. "
            "The shape parameter p is not representable there. Use cost_infinite_time(system) for "
            "the floor itself, or shorten the switching time."
        )

    def residual(p: float) -> float:
        return scale * p * float(complete_k(-(alpha**2) * p**2)) - switching_time

    lower = switching_time / (2.0 * math.pi * tau0 * (1.0 + alpha**2))
    upper = lower
    for _ in range(_MAX_BRACKET_DOUBLINGS):
        upper *= 2.0
        if residual(upper) > 0.0:
            break
    else:  # pragma: no cover
        raise RuntimeError(
            "could not bracket the shape parameter after doubling the upper end "
            f"{_MAX_BRACKET_DOUBLINGS} times; check that mu, K and gamma are in the package units"
        )

    machine = float(np.finfo(float).eps)
    relative_tolerance = 4.0 * machine  # the tightest brentq accepts
    absolute_tolerance = max(lower * relative_tolerance, float(np.finfo(float).tiny))
    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=absolute_tolerance,
            rtol=relative_tolerance,
            maxiter=_MAX_ROOT_ITERATIONS,
        )
    )


def cost_free_macrospin(switching_time: float, alpha: float, gamma: float) -> float:
    """The switching cost of a moment with **no** magnetic potential, ``Phi_f``.

    ``Phi_f = pi^2 (1 + alpha^2) / (gamma^2 T)``.

    Args:
        switching_time: ``T`` in s.
        alpha: Gilbert damping, dimensionless.
        gamma: gyromagnetic ratio in rad/(s T).

    Returns:
        Cost in T^2 s.

    Notes:
        This is the reference every material is scored against. A uniaxial magnet can never beat it
        (Phys. Rev. Lett. 126, 177206). A biaxial magnet can, and by how much is the measure of how
        usefully its internal torque assists the reversal.
    """
    return math.pi**2 * (1.0 + alpha**2) / (gamma**2 * switching_time)


def cost_infinite_time(system: MacrospinSystem) -> float:
    """The universal floor ``Phi_infinity = 4 alpha K / (gamma mu)``.

    Args:
        system: the macrospin.

    Returns:
        Cost in T^2 s. No protocol can go below this at any switching time.

    Notes:
        This is **linear in the Gilbert damping**, which in two-dimensional van der Waals magnets is
        the least well pinned parameter of the whole problem. Any single-number claim about the
        reachable energy floor in these materials is a claim about a damping nobody has measured to
        better than an order of magnitude. Report a band.
    """
    return 4.0 * system.alpha * system.anisotropy_j / (system.gamma * system.mu)


def optimal_switching_time(
    system: MacrospinSystem, tolerance: float = DEFAULT_KNEE_TOLERANCE
) -> float:
    """The switching time beyond which waiting longer buys almost no efficiency.

    ``T_eps = 2 ln(4/eps) [alpha + 1/alpha] tau0``, at which ``Phi_m = (1 + eps) Phi_infinity``.

    Args:
        system: the macrospin. Requires positive damping.
        tolerance: ``eps``, the fractional excess over the floor that is accepted. The published
            figure uses 0.1.

    Returns:
        Time in s.

    Raises:
        ValueError: if the damping is zero, where the floor is zero and the knee does not exist.
    """
    if system.alpha <= 0.0:
        raise ValueError(
            "the optimal switching time is undefined at zero damping: Phi_infinity vanishes and the "
            "cost falls as 1/T without ever reaching a knee"
        )
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie strictly between 0 and 1")
    return 2.0 * math.log(4.0 / tolerance) * (system.alpha + 1.0 / system.alpha) * system.tau0


@dataclass(frozen=True)
class UniaxialOptimalControl:
    """The exact optimal control path and pulse for a uniaxial macrospin.

    Construct with :meth:`for_switching_time`, which solves for the shape parameter, then sample the
    trajectory and the pulse on any time grid.

    Attributes:
        system: the macrospin. Its ``hard_axis_ratio`` must be zero; this solution is uniaxial.
        switching_time: ``T`` in s.
        p: the dimensionless shape parameter solving the period relation.
    """

    system: MacrospinSystem
    switching_time: float
    p: float

    @classmethod
    def for_switching_time(
        cls, system: MacrospinSystem, switching_time: float
    ) -> UniaxialOptimalControl:
        """Build the solution for a given system and switching time.

        Args:
            system: the macrospin. Must be uniaxial (``hard_axis_ratio == 0``).
            switching_time: ``T`` in s.

        Returns:
            The solution.

        Raises:
            ValueError: if the system carries a hard axis, for which no closed form exists and
                :mod:`spinoct.numeric` must be used instead.
        """
        if system.hard_axis_ratio != 0.0:
            raise ValueError(
                "the closed-form solution covers the uniaxial case only (hard_axis_ratio == 0). "
                "For a biaxial system use the numerical optimal control path solver."
            )
        return cls(
            system=system,
            switching_time=switching_time,
            p=solve_shape_parameter(switching_time, system),
        )

    # ------------------------------------------------------------------ internals

    @property
    def _m(self) -> float:
        """The elliptic parameter ``-alpha^2 p^2``, which is negative for any finite damping."""
        return -(self.system.alpha**2) * self.p**2

    @property
    def _u_scale(self) -> float:
        """The factor converting time to the elliptic argument, ``p tau0 (1 + alpha^2)`` in s."""
        return self.p * self.system.tau0 * (1.0 + self.system.alpha**2)

    def _u(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(t, dtype=float) / self._u_scale

    # ------------------------------------------------------------------ trajectory

    def polar_angle(self, t: float | np.ndarray) -> np.ndarray:
        """The polar angle ``theta(t) = am(u | m) / 2`` along the optimal control path.

        Args:
            t: time in s, scalar or array. ``t = 0`` gives 0 and ``t = T`` gives ``pi``.

        Returns:
            Angle in rad.
        """
        return 0.5 * jacobi_am(self._u(t), self._m)

    def azimuthal_angle(
        self, t: float | np.ndarray, initial_phase: float = 0.0, samples: int = 8001
    ) -> np.ndarray:
        """The azimuthal angle ``phi(t)``, from quadrature of ``tau0 phi' = cos(theta)/(1+a^2)``.

        Args:
            t: time in s, scalar or array. Values must lie in ``[0, T]``.
            initial_phase: ``phi(0)`` in rad. The uniaxial problem is axially symmetric, so this is
                free and every choice gives an equivalent optimal control path.
            samples: the size of the internal quadrature grid used for scalar or sparse queries. The
                integrand is smooth, so a few thousand points already integrate it to high accuracy.

        Returns:
            Angle in rad, same shape as ``t``.

        Notes:
            When ``t`` is itself a dense, sorted grid starting at the initial time, the cumulative
            integral is built **directly on that grid** rather than on a separate internal grid
            followed by interpolation. Interpolating a cumulative integral back onto a fine output
            grid was the accuracy bottleneck: it left a second-order error tied to the internal grid
            spacing that did not shrink as the caller refined its own grid. Integrating on the
            caller's grid removes that error, so a caller sampling the pulse and the trajectory on the
            same grid gets a self-consistent pair, which is what the inversion of the equation of
            motion relies on.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        prefactor = 1.0 / (self.system.tau0 * (1.0 + self.system.alpha**2))

        is_grid_from_origin = (
            t_arr.ndim == 1
            and t_arr.size >= 3
            and abs(float(t_arr[0])) <= _ORIGIN_RELATIVE_TOLERANCE * self.switching_time
            and bool(np.all(np.diff(t_arr) > 0.0))
        )

        if is_grid_from_origin:
            integrand = np.cos(self.polar_angle(t_arr)) * prefactor
            steps = np.diff(t_arr)
            increments = 0.5 * steps * (integrand[1:] + integrand[:-1])
            cumulative = np.concatenate(([0.0], np.cumsum(increments)))
            out = initial_phase + cumulative
        else:
            grid = np.linspace(0.0, self.switching_time, samples)
            integrand = np.cos(self.polar_angle(grid)) * prefactor
            step = grid[1] - grid[0]
            cumulative = np.concatenate(
                ([0.0], np.cumsum(0.5 * step * (integrand[1:] + integrand[:-1])))
            )
            out = initial_phase + np.interp(t_arr, grid, cumulative)

        return out if np.ndim(t) else out[0]

    def moment(self, t: float | np.ndarray, initial_phase: float = 0.0) -> np.ndarray:
        """The unit moment direction along the optimal control path.

        Args:
            t: time in s, scalar or array.
            initial_phase: ``phi(0)`` in rad.

        Returns:
            Unit vectors of shape ``(..., 3)``, with the easy axis along z, so ``s_z`` runs from
            ``+1`` at ``t = 0`` to ``-1`` at ``t = T``.
        """
        theta = np.atleast_1d(self.polar_angle(t))
        phi = np.atleast_1d(self.azimuthal_angle(t, initial_phase=initial_phase))
        out = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )
        return out if np.ndim(t) else out[0]

    # ------------------------------------------------------------------ the pulse

    def field_amplitude(self, t: float | np.ndarray) -> np.ndarray:
        """The optimal field amplitude ``b(t)``, in tesla.

        ``b(t) = K / (mu p sqrt(1+a^2)) * [ dn(u|m) + a p sn(u|m) ]``.

        Args:
            t: time in s, scalar or array.

        Returns:
            Amplitude in T.
        """
        alpha = self.system.alpha
        sn, _, dn = jacobi_snd(self._u(t), self._m)
        prefactor = self.system.anisotropy_j / (self.system.mu * self.p * math.sqrt(1.0 + alpha**2))
        return prefactor * (dn + alpha * self.p * sn)

    def peak_amplitude(self) -> float:
        """The largest field amplitude the optimal pulse demands, in tesla. Exact, not sampled.

        This is the number a driver has to be able to supply, so it is worth getting right, and the
        obvious way to get it is wrong. The amplitude is

            b(u) = K / (mu p sqrt(1+a^2)) * [dn(u|m) + a p sn(u|m)],   m = -a^2 p^2 < 0,

        and over the pulse ``u`` runs from 0 to ``4K(m)`` (the polar angle is half the Jacobi
        amplitude, which reaches ``2 pi`` there). At the start, the midpoint and the end ``sn = 0`` and
        ``dn = 1``, so those three points share one middle value, which is neither extremum. For a
        negative parameter ``dn^2 = 1 - m sn^2 >= 1``, so the amplitude is largest where ``sn = 1``, at
        ``u = K(m)``, a quarter of the way through, with ``dn = sqrt(1 - m) = sqrt(1 + a^2 p^2)``, and
        smallest where ``sn = -1``, at three quarters (see :meth:`peak_times`). Sampling the start and
        the midpoint, the obvious shortcut, reports that middle value: up to 27 per cent below the peak
        at ``alpha = 0.1`` and ``T = 20 tau0`` (the peak is 37 per cent above it), and the gap grows with
        damping and with the switching time.

        Returns:
            ``K / (mu p sqrt(1+a^2)) * [sqrt(1 + a^2 p^2) + a p]``, in T. Agrees with a 200,001-point
            scan of :meth:`field_amplitude` to machine precision over alpha in [0.001, 0.3] and T in
            [0.5, 50] tau0.
        """
        alpha = self.system.alpha
        prefactor = self.system.anisotropy_j / (self.system.mu * self.p * math.sqrt(1.0 + alpha**2))
        return float(prefactor * (math.sqrt(1.0 + (alpha * self.p) ** 2) + alpha * self.p))

    def field_vector(self, t: float | np.ndarray, initial_phase: float = 0.0) -> np.ndarray:
        """The optimal field as a Cartesian vector, in tesla.

        The field lies in the local frame as ``b/sqrt(1+a^2) * (alpha e_theta + e_phi)``, which is
        always perpendicular to the moment. That perpendicularity is the reason the longitudinal
        stabilizing field studied in arXiv:2312.11293 is invisible to the leading-order dynamics
        while still costing energy.

        Args:
            t: time in s, scalar or array.
            initial_phase: ``phi(0)`` in rad.

        Returns:
            Field vectors of shape ``(..., 3)`` in T.
        """
        alpha = self.system.alpha
        theta = np.atleast_1d(self.polar_angle(t))
        phi = np.atleast_1d(self.azimuthal_angle(t, initial_phase=initial_phase))
        amplitude = np.atleast_1d(self.field_amplitude(t))

        e_theta = np.stack(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)], axis=-1
        )
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
        out = (amplitude / math.sqrt(1.0 + alpha**2))[..., None] * (alpha * e_theta + e_phi)
        return out if np.ndim(t) else out[0]

    # ------------------------------------------------------------------ closed forms

    def cost(self) -> float:
        """The minimum switching cost ``Phi_m``, in T^2 s.

        ``Phi_m = 2 K [2 E(m) - K(m)] / (gamma mu p)`` with ``m = -alpha^2 p^2``.

        Returns:
            Cost in T^2 s.
        """
        m = self._m
        numerator = 2.0 * self.system.anisotropy_j * (2.0 * float(complete_e(m)) - float(complete_k(m)))
        return numerator / (self.system.gamma * self.system.mu * self.p)

    def mean_amplitude(self) -> float:
        """The time-averaged field amplitude, ``pi sqrt(1+a^2) / (gamma T)``, in tesla.

        Returns:
            Amplitude in T.

        Notes:
            Independent of the magnetic potential, which is why it is a clean check that the
            trajectory and the pulse were built consistently.
        """
        return math.pi * math.sqrt(1.0 + self.system.alpha**2) / (self.system.gamma * self.switching_time)

    def amplitude_spread(self) -> float:
        """The exact peak-to-trough spread ``2 alpha K / (mu sqrt(1+a^2))``, in tesla.

        Returns:
            The spread in T. Zero at zero damping, where the optimal amplitude is constant.
        """
        alpha = self.system.alpha
        return 2.0 * alpha * self.system.anisotropy_j / (self.system.mu * math.sqrt(1.0 + alpha**2))

    def peak_times(self) -> tuple[float, float]:
        """The exact times of the amplitude maximum and minimum.

        Returns:
            ``(t_max, t_min)`` in s, which are ``T/4`` and ``3T/4``. The extrema sit where ``cn``
            vanishes, and ``u(T) = 4 K(m)`` maps those to exactly the quarter and three-quarter
            points of the switching window.
        """
        return self.switching_time / 4.0, 3.0 * self.switching_time / 4.0

    def cost_ratio_to_free(self) -> float:
        """``Phi_m / Phi_f``, the cost relative to a moment with no magnetic potential.

        Returns:
            Dimensionless. For a uniaxial system this is at least 1, with equality only at zero
            damping: easy-axis anisotropy can only obstruct the reversal.
        """
        return self.cost() / cost_free_macrospin(
            self.switching_time, self.system.alpha, self.system.gamma
        )

    def cost_ratio_to_floor(self) -> float:
        """``Phi_m / Phi_infinity``, the cost relative to the universal long-time floor.

        Returns:
            Dimensionless, at least 1.

        Raises:
            ValueError: at zero damping, where the floor vanishes.
        """
        floor = cost_infinite_time(self.system)
        if floor <= 0.0:
            raise ValueError("the infinite-time floor vanishes at zero damping; the ratio is undefined")
        return self.cost() / floor
