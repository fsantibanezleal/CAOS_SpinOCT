"""The spin-orbit-torque couplings, converted from the Gilbert form the literature writes them in."""

from __future__ import annotations

__all__ = ["explicit_sot_coefficients"]


def explicit_sot_coefficients(xi_f: float, xi_d: float, alpha: float) -> tuple[float, float]:
    """The field-like and damping-like coefficients of the EXPLICIT equation of motion.

    The source (Vlasov et al., Phys. Rev. B 105, 134404, Eq. 3) writes the spin-orbit torques inside
    the implicit Gilbert form, ``s_dot = tau + alpha s x s_dot``. Solving it for a unit moment gives
    ``(1 + alpha^2) s_dot = tau + alpha s x tau``, and the cross product mixes the two torque channels:
    ``alpha s x (xi_F s x p)`` is damping-like and ``alpha s x (xi_D s x (s x p)) = -alpha xi_D s x p``
    is field-like. So the explicit coefficients, each over ``1 + alpha^2``, are

        field-like:    xi_F - alpha xi_D
        damping-like:  xi_D + alpha xi_F

    The paper's sweet spot is the check: at its ideal ratio ``xi_D = -alpha xi_F`` the explicit
    damping-like term vanishes, which is why it says the problem "becomes identical to" field-driven
    switching, and at its forbidden ratio ``xi_F = alpha xi_D`` the field-like term vanishes.

    An earlier version of this integrator used ``xi_F`` and ``xi_D`` directly as the explicit
    coefficients while its docstring stated the Gilbert form, so it solved a different equation from
    the one it cited: 2 to 22 per cent off at ``alpha = 0.1``, most where the damping-like coupling
    dominates (measured 2026-09-18, against a direct linear solve of the implicit equation).
    """
    return xi_f - alpha * xi_d, xi_d + alpha * xi_f
