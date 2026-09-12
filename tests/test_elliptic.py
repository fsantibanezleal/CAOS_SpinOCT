"""The negative-parameter Jacobi elliptic functions, checked against their defining integral.

The imaginary-modulus transformation in ``spinoct.analytic.elliptic`` is a closed-form identity, so
testing it against itself proves nothing. Every test here goes back to the definition

    u = int_0^phi dtheta / sqrt(1 - m sin^2 theta),   sn = sin phi,  cn = cos phi,
    dn = sqrt(1 - m sin^2 phi),  am = phi

which is perfectly well defined for negative ``m`` (the integrand stays real and bounded), and is
evaluated here by quadrature and root finding rather than by any elliptic identity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from spinoct.analytic.elliptic import (
    complete_e,
    complete_k,
    jacobi_am,
    jacobi_cn,
    jacobi_dn,
    jacobi_sn,
)

NEGATIVE_PARAMETERS = [-0.01, -0.25, -1.0, -4.0, -25.0]


def _incomplete_first_kind(phi: float, m: float) -> float:
    """``F(phi|m)`` by direct quadrature, independent of any elliptic identity."""
    value, _ = quad(lambda th: 1.0 / math.sqrt(1.0 - m * math.sin(th) ** 2), 0.0, phi, limit=200)
    return value


def _amplitude_by_inversion(u: float, m: float) -> float:
    """Invert ``F(phi|m) = u`` numerically to get the amplitude, for any real ``u`` and ``m < 1``."""
    lower, upper = -1.0, 1.0
    while _incomplete_first_kind(upper, m) < u:
        upper *= 2.0
    while _incomplete_first_kind(lower, m) > u:
        lower *= 2.0
    return brentq(lambda p: _incomplete_first_kind(p, m) - u, lower, upper, xtol=1e-14)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
@pytest.mark.parametrize("u", [0.0, 0.3, 1.0, 2.5, 5.0, 9.0])
def test_amplitude_matches_direct_inversion(u: float, m: float) -> None:
    expected = _amplitude_by_inversion(u, m)
    assert jacobi_am(u, m) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
@pytest.mark.parametrize("u", [0.0, 0.3, 1.0, 2.5, 5.0, 9.0])
def test_sn_cn_dn_match_the_definition(u: float, m: float) -> None:
    phi = _amplitude_by_inversion(u, m)
    assert jacobi_sn(u, m) == pytest.approx(math.sin(phi), abs=1e-10)
    assert jacobi_cn(u, m) == pytest.approx(math.cos(phi), abs=1e-10)
    assert jacobi_dn(u, m) == pytest.approx(math.sqrt(1.0 - m * math.sin(phi) ** 2), abs=1e-10)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
def test_pythagorean_identities(m: float) -> None:
    u = np.linspace(-8.0, 8.0, 401)
    sn, cn, dn = jacobi_sn(u, m), jacobi_cn(u, m), jacobi_dn(u, m)
    np.testing.assert_allclose(sn**2 + cn**2, 1.0, atol=1e-12)
    np.testing.assert_allclose(m * sn**2 + dn**2, 1.0, atol=1e-12)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
def test_dn_is_at_least_one_for_negative_parameter(m: float) -> None:
    """For ``m < 0`` the familiar ``dn <= 1`` is reversed, which is easy to get wrong by analogy."""
    u = np.linspace(-8.0, 8.0, 401)
    assert np.all(jacobi_dn(u, m) >= 1.0 - 1e-12)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
def test_amplitude_is_unwrapped_and_monotone(m: float) -> None:
    """``am`` must keep increasing past ``pi/2`` rather than folding, or theta never reaches pi."""
    u = np.linspace(0.0, 6.0 * float(complete_k(m)), 2001)
    am = jacobi_am(u, m)
    assert np.all(np.diff(am) > 0.0)
    assert am[0] == pytest.approx(0.0, abs=1e-12)
    assert jacobi_am(4.0 * float(complete_k(m)), m) == pytest.approx(2.0 * math.pi, abs=1e-10)


@pytest.mark.parametrize("m", NEGATIVE_PARAMETERS)
def test_amplitude_advances_by_pi_every_half_period(m: float) -> None:
    quarter = float(complete_k(m))
    for u in (0.37, 1.9, 4.2):
        assert jacobi_am(u + 2.0 * quarter, m) == pytest.approx(jacobi_am(u, m) + math.pi, abs=1e-10)


def test_zero_parameter_reduces_to_trigonometry() -> None:
    u = np.linspace(-6.0, 6.0, 301)
    np.testing.assert_allclose(jacobi_sn(u, 0.0), np.sin(u), atol=1e-12)
    np.testing.assert_allclose(jacobi_cn(u, 0.0), np.cos(u), atol=1e-12)
    np.testing.assert_allclose(jacobi_dn(u, 0.0), 1.0, atol=1e-12)
    np.testing.assert_allclose(jacobi_am(u, 0.0), u, atol=1e-12)


@pytest.mark.parametrize("m", [*NEGATIVE_PARAMETERS, 0.0, 0.5])
def test_complete_integrals_match_quadrature(m: float) -> None:
    expected_k, _ = quad(lambda th: 1.0 / math.sqrt(1.0 - m * math.sin(th) ** 2), 0.0, math.pi / 2)
    expected_e, _ = quad(lambda th: math.sqrt(1.0 - m * math.sin(th) ** 2), 0.0, math.pi / 2)
    assert float(complete_k(m)) == pytest.approx(expected_k, rel=1e-12)
    assert float(complete_e(m)) == pytest.approx(expected_e, rel=1e-12)
