"""The minimum energy path of a spin chain, and the energy floor it puts under every switching pulse.

Why this module exists
----------------------
The switching cost has a lower bound that holds at every switching time and for every trajectory, and
it is set by an energy barrier. Invert the equation of motion at one site,

    b = (alpha / gamma) s_dot + (1 / gamma) s x s_dot - b_int_perp,

and dot it with ``s_dot``. The internal field does work at the rate ``-mu b_int . s_dot``, which is the
rate of change of the site energy, so

    mu b . s_dot = dE/dt + mu (alpha / gamma) |s_dot|^2.

Only the component of ``b`` along ``s_dot`` enters, so ``|b|^2 >= (b . s_dot_hat)^2``. On any stretch
where the energy rises, the arithmetic-geometric mean inequality applied to the two terms gives

    |b|^2 >= (4 alpha / (gamma mu)) dE/dt.

Summed over sites and integrated over the rise, every reversal therefore costs at least

    Phi >= (4 alpha / (gamma mu)) Delta E,

where ``Delta E`` is the largest energy rise the path makes, which is at least the barrier of the
MINIMUM energy path between the two states. For a single uniaxial site ``Delta E = K`` and the bound is
exactly the known infinite-time optimum ``Phi_inf = 4 alpha K / (gamma mu)`` (Kwiatkowski et al.,
Phys. Rev. Lett. 126, 177206, 2021), so the bound is tight there. For a chain, uniform rotation has
barrier ``N K``, while a path that nucleates a domain wall at an open end and sweeps it through can
have a much lower barrier. If the free optimal control path of a long chain is cheaper than uniform
rotation, this is the mechanism, and the MEP barrier predicts the long-time saving.

Method
------
The string method (W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys. 126, 164103, 2007) on the product
of unit spheres, in the geodesic form used for magnetic systems (P. F. Bessarab, V. M. Uzdin,
H. Jonsson, Comput. Phys. Commun. 196, 335, 2015). Interior images descend the energy along the
tangent-projected force with the component along the path removed, then are redistributed to equal
geodesic spacing. A climbing image then climbs along the path tangent so the highest image converges
onto the saddle point, which gives the barrier accurately rather than to within the image spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import SpinChain

__all__ = ["MinimumEnergyPath", "cost_floor_from_barrier", "minimum_energy_path"]

#: Relative step on the dimensionless force (force in units of the anisotropy energy per site).
_STEP = 0.02


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _energies(chain: SpinChain, images: np.ndarray) -> np.ndarray:
    anisotropy = -chain.anisotropy_j * np.sum(images[..., 2] ** 2, axis=-1)
    exchange = -chain.exchange_j * np.sum(np.sum(images[:, :-1] * images[:, 1:], axis=-1), axis=-1)
    return anisotropy + exchange


def _forces(chain: SpinChain, images: np.ndarray) -> np.ndarray:
    """``-dE/ds`` projected onto each site's tangent plane, in J per unit vector."""
    field = np.zeros_like(images)
    field[..., 2] += 2.0 * chain.anisotropy_j * images[..., 2]
    field[:, :-1] += chain.exchange_j * images[:, 1:]
    field[:, 1:] += chain.exchange_j * images[:, :-1]
    return field - np.sum(field * images, axis=-1, keepdims=True) * images


def _geodesic_lengths(images: np.ndarray) -> np.ndarray:
    """Distance between consecutive images on the product of spheres, shape ``(M - 1,)``."""
    left, right = images[:-1], images[1:]
    angles = 2.0 * np.arctan2(
        np.linalg.norm(right - left, axis=-1), np.linalg.norm(right + left, axis=-1)
    )
    return np.sqrt(np.sum(angles**2, axis=-1))


def _redistribute(images: np.ndarray) -> np.ndarray:
    """Re-space interior images to equal geodesic arc length by per-site spherical interpolation."""
    lengths = _geodesic_lengths(images)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], images.shape[0])
    out = images.copy()
    for m in range(1, images.shape[0] - 1):
        k = int(np.clip(np.searchsorted(cumulative, targets[m]) - 1, 0, images.shape[0] - 2))
        span = cumulative[k + 1] - cumulative[k]
        w = 0.0 if span <= 0.0 else (targets[m] - cumulative[k]) / span
        out[m] = _normalize((1.0 - w) * images[k] + w * images[k + 1])
    return out


def _tangents(images: np.ndarray) -> np.ndarray:
    """Unit path tangents at interior images (central chord, projected per site), shape like images."""
    tangent = np.zeros_like(images)
    chord = images[2:] - images[:-2]
    chord = chord - np.sum(chord * images[1:-1], axis=-1, keepdims=True) * images[1:-1]
    norms = np.sqrt(np.sum(chord**2, axis=(-1, -2), keepdims=True))
    tangent[1:-1] = np.divide(chord, norms, out=np.zeros_like(chord), where=norms > 0.0)
    return tangent


@dataclass(frozen=True)
class MinimumEnergyPath:
    """A converged minimum energy path.

    Attributes:
        images: the path, shape ``(M, N, 3)``, from all up to all down.
        energies: the energy of each image relative to all up, J.
        barrier: the saddle-point energy relative to all up, J.
        saddle_index: the index of the climbing image.
        converged: whether the force criterion was met.
        iterations: steps taken.
    """

    images: np.ndarray
    energies: np.ndarray
    barrier: float
    saddle_index: int
    converged: bool
    iterations: int

    def barrier_over_uniform(self, chain: SpinChain) -> float:
        """``barrier / (N K)``: the MEP barrier relative to the uniform-rotation barrier."""
        return self.barrier / (chain.n_sites * chain.anisotropy_j)


def _initial_path(chain: SpinChain, n_images: int, mode: str) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, n_images)
    n = chain.n_sites
    if mode == "uniform":
        theta = np.repeat((np.pi * fractions)[:, None], n, axis=1)
    elif mode == "wall":
        width = max(1.0, float(np.sqrt(chain.exchange_j / (2.0 * chain.anisotropy_j))))
        centre = -2.0 * width + (n - 1 + 4.0 * width) * fractions
        progress = 0.5 * (1.0 + np.tanh((centre[:, None] - np.arange(n)[None, :]) / width))
        progress = (progress - progress[0]) / (progress[-1] - progress[0])
        theta = np.pi * progress
    else:
        raise ValueError("mode must be 'uniform' or 'wall'")
    # The energy is invariant under a common rotation about z, so the path can live in the x-z plane.
    return np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)


def minimum_energy_path(
    chain: SpinChain,
    n_images: int = 33,
    initial: str = "wall",
    max_iterations: int = 20000,
    force_tolerance: float = 1e-6,
) -> MinimumEnergyPath:
    """Find the minimum energy path from all up to all down by the climbing-image string method.

    Args:
        chain: the spin chain.
        n_images: images along the path, including both clamped ends.
        initial: ``"wall"`` (a wall entering at site 0) or ``"uniform"`` (coherent rotation).
        max_iterations: iteration cap.
        force_tolerance: stop when the largest perpendicular force on any interior image, in units of
            the anisotropy energy per site, falls below this (dimensionless).

    Returns:
        The :class:`MinimumEnergyPath`.
    """
    if n_images < 3:
        raise ValueError("n_images must be at least 3")
    images = _initial_path(chain, n_images, initial)
    scale = chain.anisotropy_j
    converged = False
    iterations = 0
    climbing = False
    for _ in range(max_iterations):
        iterations += 1
        forces = _forces(chain, images) / scale
        tangent = _tangents(images)
        along = np.sum(forces * tangent, axis=(-1, -2), keepdims=True)
        perpendicular = forces - along * tangent
        update = perpendicular.copy()
        energies = _energies(chain, images)
        top = int(np.argmax(energies[1:-1])) + 1
        if climbing:
            # The climbing image inverts its force along the path and so converges onto the saddle.
            update[top] = forces[top] - 2.0 * along[top] * tangent[top]
        update[0] = 0.0
        update[-1] = 0.0
        residual = float(np.max(np.linalg.norm(update[1:-1], axis=-1)))
        if residual < force_tolerance:
            if climbing:
                converged = True
                break
            climbing = True
        images = _normalize(images + _STEP * update)
        if climbing:
            saved = images[top].copy()
            images = _redistribute(images)
            images[top] = saved
        else:
            images = _redistribute(images)

    energies = _energies(chain, images) - _energies(chain, images[:1])[0]
    top = int(np.argmax(energies))
    return MinimumEnergyPath(
        images=images,
        energies=energies,
        barrier=float(energies[top]),
        saddle_index=top,
        converged=converged,
        iterations=iterations,
    )


def cost_floor_from_barrier(barrier_j: float, mu: float, alpha: float, gamma: float) -> float:
    """The switching-cost floor ``4 alpha Delta E / (gamma mu)`` implied by an energy barrier, T^2 s.

    Args:
        barrier_j: the barrier ``Delta E``, J.
        mu: the moment per site, J/T.
        alpha: Gilbert damping.
        gamma: gyromagnetic ratio, rad/(s T).

    Returns:
        The floor in T^2 s.
    """
    return 4.0 * alpha * barrier_j / (gamma * mu)
