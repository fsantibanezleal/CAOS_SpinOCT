"""Stochastic Landau-Lifshitz-Gilbert dynamics with a thermal field, and the switching success rate.

The thermal field
-----------------
At temperature the moment feels a fluctuating field whose strength is fixed by the
fluctuation-dissipation theorem: the same damping that dissipates energy also injects thermal noise.
For a single macrospin the white-noise field has the covariance

    <b_th_i(t) b_th_j(t')> = (2 alpha k_B T) / (gamma mu) delta_ij delta(t - t')

so a time step of length dt draws each Cartesian component from a Gaussian of variance
2 alpha k_B T / (gamma mu dt). This is the standard atomistic-spin-dynamics thermostat (Evans et al.,
J. Phys.: Condens. Matter 26, 103202 (2014), https://doi.org/10.1088/0953-8984/26/10/103202).

Integration is by the stochastic Heun scheme (a predictor-corrector), which converges to the
Stratonovich solution the physics requires, and the moment is renormalized each step so it stays on the
sphere. The zero-temperature limit reduces to the deterministic dynamics of :mod:`spinoct.dynamics`.

The success rate
----------------
Following the protocol of Badarneh, Kwiatkowski and Bessarab, Phys. Rev. B 107, 214448 (2023): for
each of many independent copies, apply the switching field during the window with thermal fluctuations
on, then check whether the moment ended in the reversed basin (below a threshold on s_z). The fraction
that switched is the success rate, the reliability metric a memory element must keep near unity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..dynamics.system import MacrospinSystem
from ..units import thermal_energy_j

__all__ = [
    "EnsembleResult",
    "boltzmann_polar_variance",
    "stochastic_llg_step",
    "switching_success_rate",
]


def _deterministic_rhs(
    s: np.ndarray, b_total: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    """The LLG right-hand side for a batch of moments under a total field, shape ``(..., 3)``."""
    precession = np.cross(s, b_total)
    damping = np.cross(s, precession)
    return (-gamma * precession - alpha * gamma * damping) / (1.0 + alpha**2)


def thermal_field_scale(system: MacrospinSystem, temperature_k: float, dt: float) -> float:
    """The standard deviation of each thermal-field component over a step, in tesla.

    Args:
        system: the macrospin.
        temperature_k: temperature, K.
        dt: the time step, s.

    Returns:
        The per-component standard deviation, T. Zero at zero temperature.
    """
    if temperature_k <= 0.0:
        return 0.0
    numerator = 2.0 * system.alpha * thermal_energy_j(temperature_k)
    return float(np.sqrt(numerator / (system.gamma * system.mu * dt)))


def stochastic_llg_step(
    s: np.ndarray,
    applied_field: np.ndarray,
    system: MacrospinSystem,
    temperature_k: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance a batch of moments one stochastic Heun step.

    Args:
        s: unit moments, shape ``(N, 3)``.
        applied_field: the deterministic applied field per moment, shape ``(N, 3)`` or ``(3,)``, T.
        system: the macrospin.
        temperature_k: temperature, K.
        dt: the time step, s.
        rng: the random generator, for reproducibility.

    Returns:
        The advanced, renormalized moments, shape ``(N, 3)``.
    """
    s = np.asarray(s, dtype=float)
    applied = np.broadcast_to(np.asarray(applied_field, dtype=float), s.shape)
    alpha, gamma = system.alpha, system.gamma

    scale = thermal_field_scale(system, temperature_k, dt)
    noise = rng.normal(scale=scale, size=s.shape) if scale > 0.0 else np.zeros_like(s)

    def total_field(moment: np.ndarray) -> np.ndarray:
        return system.internal_field(moment) + applied + noise

    # Heun predictor.
    k1 = _deterministic_rhs(s, total_field(s), alpha, gamma)
    predictor = s + dt * k1
    predictor = predictor / np.linalg.norm(predictor, axis=-1, keepdims=True)
    # Heun corrector, same noise sample (the Stratonovich convention).
    k2 = _deterministic_rhs(predictor, total_field(predictor), alpha, gamma)
    out = s + 0.5 * dt * (k1 + k2)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def boltzmann_polar_variance(system: MacrospinSystem, temperature_k: float) -> float:
    """The small-angle variance of the polar angle at equilibrium in a minimum, radians squared.

    Near a uniaxial minimum the energy is ``E ~ K theta^2`` (with ``s_z = cos theta``, so
    ``-K s_z^2 ~ -K + K theta^2``), so by equipartition in two transverse directions the mean square
    polar angle is ``<theta^2> = k_B T / K``.

    Args:
        system: the macrospin.
        temperature_k: temperature, K.

    Returns:
        The variance in radians squared. Used to check the thermostat reproduces the Boltzmann
        distribution.
    """
    return thermal_energy_j(temperature_k) / system.anisotropy_j


@dataclass(frozen=True)
class EnsembleResult:
    """The outcome of a switching-success ensemble.

    Attributes:
        success_rate: the fraction of copies that ended in the reversed basin.
        n_copies: the ensemble size.
        thermal_stability_factor: ``Delta = K / (k_B T)``, the barrier over the thermal energy.
        final_sz_mean: the mean final z-component over the ensemble.
        confidence95: the half-width of the 95 percent binomial confidence interval on the rate.
    """

    success_rate: float
    n_copies: int
    thermal_stability_factor: float
    final_sz_mean: float
    confidence95: float


def switching_success_rate(
    system: MacrospinSystem,
    field: Callable[[float], np.ndarray],
    switching_time: float,
    temperature_k: float,
    n_copies: int = 500,
    n_steps: int = 800,
    threshold: float = -0.5,
    seed: int = 0,
) -> EnsembleResult:
    """Estimate the switching success rate at temperature over an ensemble.

    Args:
        system: the macrospin.
        field: the applied field as a function of time, shape ``(3,)``, T.
        switching_time: the window, s.
        temperature_k: temperature, K.
        n_copies: the ensemble size.
        n_steps: integration steps across the window.
        threshold: the final ``s_z`` below which a copy counts as switched.
        seed: the random seed.

    Returns:
        The :class:`EnsembleResult`.
    """
    rng = np.random.default_rng(seed)
    dt = switching_time / n_steps
    times = np.linspace(0.0, switching_time, n_steps + 1)

    # Every copy starts at the north pole; a short thermal pre-equilibration would only add a small
    # initial spread, which at the stability factors of interest does not change the rate.
    s = np.tile(np.array([0.0, 0.0, 1.0]), (n_copies, 1))
    field_table = np.stack([field(float(t)) for t in times])

    for index in range(n_steps):
        b_mid = 0.5 * (field_table[index] + field_table[index + 1])
        s = stochastic_llg_step(s, b_mid, system, temperature_k, dt, rng)

    final_sz = s[:, 2]
    switched = final_sz < threshold
    rate = float(np.mean(switched))
    # Wald 95 percent interval half-width, floored so a degenerate 0 or 1 still reports a width.
    variance = max(rate * (1.0 - rate), 1.0 / n_copies)
    half_width = 1.96 * float(np.sqrt(variance / n_copies))

    return EnsembleResult(
        success_rate=rate,
        n_copies=n_copies,
        thermal_stability_factor=system.thermal_stability_factor(temperature_k),
        final_sz_mean=float(np.mean(final_sz)),
        confidence95=half_width,
    )
