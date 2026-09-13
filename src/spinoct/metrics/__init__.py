"""The domain metric suite for scoring switching protocols.

Every protocol, optimal or conventional, is scored on the same quantities so a comparison is
apples-to-apples. The metrics are dimensionless where possible, because a dimensionless ratio is
comparable across materials while a raw cost is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..analytic.uniaxial import cost_free_macrospin, cost_infinite_time
from ..dynamics.system import MacrospinSystem

__all__ = ["ProtocolMetrics", "pulse_bandwidth_fraction", "spectral_energy_fraction"]


def spectral_energy_fraction(
    times: np.ndarray, amplitude: np.ndarray, fraction: float = 0.99
) -> float:
    """The frequency below which a given fraction of the pulse energy lies, in hertz.

    Args:
        times: uniform time grid, s.
        amplitude: the scalar field amplitude at each time, T.
        fraction: the energy fraction (0 to 1).

    Returns:
        The bandwidth in Hz. This is the realizability metric: a pulse whose energy sits at very high
        frequency cannot be produced by a real generator, which is why the constrained solvers exist.
    """
    times = np.asarray(times, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if times.size < 2:
        return 0.0
    dt = float(times[1] - times[0])
    spectrum = np.abs(np.fft.rfft(amplitude)) ** 2
    freqs = np.fft.rfftfreq(times.size, d=dt)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(spectrum) / total
    index = int(np.searchsorted(cumulative, fraction))
    index = min(index, freqs.size - 1)
    return float(freqs[index])


def pulse_bandwidth_fraction(times: np.ndarray, field: np.ndarray, fraction: float = 0.99) -> float:
    """The 99 percent energy bandwidth of a vector field pulse, in hertz.

    Args:
        times: uniform time grid, s.
        field: the field vector at each time, shape ``(N, 3)``, T.
        fraction: the energy fraction.

    Returns:
        The bandwidth in Hz, taken as the maximum over the three components.
    """
    field = np.asarray(field, dtype=float)
    if field.ndim == 1:
        return spectral_energy_fraction(times, field, fraction)
    return max(spectral_energy_fraction(times, field[:, k], fraction) for k in range(field.shape[1]))


@dataclass(frozen=True)
class ProtocolMetrics:
    """The scored metrics of a switching protocol.

    Attributes:
        cost: the switching cost, T^2 s.
        cost_over_floor: cost relative to the universal long-time floor, dimensionless.
        cost_over_free: cost relative to the free-macrospin cost, dimensionless. Below 1 means the
            material's internal dynamics is paying for part of the reversal.
        peak_amplitude: the largest field magnitude in the pulse, T.
        bandwidth_hz: the 99 percent energy bandwidth of the pulse, Hz.
        switched: whether the reversal completed.
        final_sz: the final z-component of the moment.
    """

    cost: float
    cost_over_floor: float
    cost_over_free: float
    peak_amplitude: float
    bandwidth_hz: float
    switched: bool
    final_sz: float

    def as_dict(self) -> dict[str, float | bool]:
        """A flat serializable view, for writing into an artifact."""
        return asdict(self)

    @classmethod
    def score(
        cls,
        system: MacrospinSystem,
        switching_time: float,
        times: np.ndarray,
        field: np.ndarray,
        cost: float,
        final_sz: float,
    ) -> ProtocolMetrics:
        """Score a protocol from its pulse and outcome.

        Args:
            system: the macrospin.
            switching_time: ``T`` in s.
            times: the time grid, s.
            field: the field vector at each time, shape ``(N, 3)``, T.
            cost: the switching cost, T^2 s.
            final_sz: the final z-component of the moment.

        Returns:
            The :class:`ProtocolMetrics`.
        """
        field = np.asarray(field, dtype=float)
        magnitude = np.linalg.norm(field, axis=-1) if field.ndim == 2 else np.abs(field)
        floor = cost_infinite_time(system)
        free = cost_free_macrospin(switching_time, system.alpha, system.gamma)
        return cls(
            cost=cost,
            cost_over_floor=cost / floor if floor > 0.0 else float("inf"),
            cost_over_free=cost / free,
            peak_amplitude=float(np.max(magnitude)),
            bandwidth_hz=pulse_bandwidth_fraction(times, field),
            switched=final_sz < 0.0,
            final_sz=final_sz,
        )
