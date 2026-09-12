"""Jacobi elliptic functions for a NEGATIVE parameter, which is the regime this package needs.

The analytic optimal control path of a uniaxial macrospin is expressed through Jacobi elliptic
functions evaluated at parameter ``m = -alpha^2 p^2``, which is negative for any finite damping
(Kwiatkowski, Badarneh, Berkov and Bessarab, Phys. Rev. Lett. 126, 177206 (2021),
https://doi.org/10.1103/PhysRevLett.126.177206).

``scipy.special.ellipj`` accepts only ``0 <= m <= 1``, so the negative-parameter case is reduced to
the positive one by the imaginary-modulus transformation (Abramowitz and Stegun, Handbook of
Mathematical Functions, section 16.10). With

    v = u * sqrt(1 - m),    mu = -m / (1 - m)   (so 0 < mu < 1 whenever m < 0),

    sn(u|m) = sn(v|mu) / [ dn(v|mu) * sqrt(1 - m) ]
    cn(u|m) = cn(v|mu) / dn(v|mu)
    dn(u|m) = 1 / dn(v|mu)

The amplitude ``am`` is recovered with its winding number rather than from a principal-branch
arctangent, because the optimal control path needs ``am`` to run monotonically from 0 to ``2 pi``
over the switching window. ``am(u + 2K(m) | m) = am(u|m) + pi`` supplies the winding.

The complete integrals ``K(m)`` and ``E(m)`` are taken straight from ``scipy.special.ellipk`` and
``ellipe``, which are defined for all ``m < 1`` including negative values.

Every function here is validated in ``tests/test_elliptic.py`` against a direct numerical inversion
of the defining incomplete integral, which is independent of the transformation above.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipj, ellipk

__all__ = ["complete_e", "complete_k", "jacobi_am", "jacobi_cn", "jacobi_dn", "jacobi_sn", "jacobi_snd"]


def complete_k(m: float | np.ndarray) -> np.ndarray:
    """Complete elliptic integral of the first kind ``K(m)``, parameter convention.

    Args:
        m: the parameter (not the modulus), any value strictly below 1.

    Returns:
        ``K(m)``.
    """
    return np.asarray(ellipk(m))


def complete_e(m: float | np.ndarray) -> np.ndarray:
    """Complete elliptic integral of the second kind ``E(m)``, parameter convention.

    Args:
        m: the parameter (not the modulus), any value at most 1.

    Returns:
        ``E(m)``.
    """
    return np.asarray(ellipe(m))


def _reduce_negative(u: np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``sn``, ``cn``, ``dn`` at a possibly negative parameter.

    Args:
        u: argument.
        m: parameter, any value strictly below 1.

    Returns:
        ``(sn, cn, dn)`` at ``(u, m)``.
    """
    u = np.asarray(u, dtype=float)
    if m >= 0.0:
        sn, cn, dn, _ = ellipj(u, m)
        return sn, cn, dn
    one_minus_m = 1.0 - m
    root = np.sqrt(one_minus_m)
    mu = -m / one_minus_m
    sn_v, cn_v, dn_v, _ = ellipj(u * root, mu)
    return sn_v / (dn_v * root), cn_v / dn_v, 1.0 / dn_v


def jacobi_sn(u: float | np.ndarray, m: float) -> np.ndarray:
    """Jacobi ``sn(u|m)``, valid for negative ``m``.

    Args:
        u: argument.
        m: parameter, strictly below 1.

    Returns:
        ``sn(u|m)``.
    """
    return _reduce_negative(np.asarray(u, dtype=float), m)[0]


def jacobi_cn(u: float | np.ndarray, m: float) -> np.ndarray:
    """Jacobi ``cn(u|m)``, valid for negative ``m``.

    Args:
        u: argument.
        m: parameter, strictly below 1.

    Returns:
        ``cn(u|m)``.
    """
    return _reduce_negative(np.asarray(u, dtype=float), m)[1]


def jacobi_dn(u: float | np.ndarray, m: float) -> np.ndarray:
    """Jacobi ``dn(u|m)``, valid for negative ``m``.

    Args:
        u: argument.
        m: parameter, strictly below 1.

    Returns:
        ``dn(u|m)``. For ``m < 0`` this is at least 1, unlike the familiar ``m > 0`` case.
    """
    return _reduce_negative(np.asarray(u, dtype=float), m)[2]


def jacobi_snd(u: float | np.ndarray, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All three of ``sn``, ``cn``, ``dn`` in one evaluation.

    Args:
        u: argument.
        m: parameter, strictly below 1.

    Returns:
        ``(sn, cn, dn)``. Preferred over three separate calls inside a solver loop.
    """
    return _reduce_negative(np.asarray(u, dtype=float), m)


def jacobi_am(u: float | np.ndarray, m: float) -> np.ndarray:
    """Jacobi amplitude ``am(u|m)``, unwrapped, valid for negative ``m``.

    The amplitude is returned with its winding number, so it increases monotonically without bound
    rather than folding into ``[-pi/2, pi/2]``. This matters: the optimal control path is
    ``theta(t) = am(...) / 2`` and must run from 0 to ``pi`` across the switching window, which
    requires ``am`` to run from 0 to ``2 pi``.

    Args:
        u: argument.
        m: parameter, strictly below 1.

    Returns:
        ``am(u|m)`` in radians, unwrapped.
    """
    u = np.asarray(u, dtype=float)
    quarter = float(complete_k(m))
    half_period = 2.0 * quarter
    winding = np.floor((u + quarter) / half_period)
    u_reduced = u - winding * half_period
    sn, cn, _ = _reduce_negative(u_reduced, m)
    return winding * np.pi + np.arctan2(sn, cn)
