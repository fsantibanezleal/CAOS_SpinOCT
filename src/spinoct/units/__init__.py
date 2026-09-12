"""Physical constants, unit conversions, and the dimensional contract of spinoct.

Why this module exists at all
----------------------------
The optimal-control literature for magnetization switching writes the same physical quantity four
different ways. Exchange constants appear as ``+sum J S.S``, ``-sum J S.S``, with and without the
one-half double-counting factor, and over unit vectors rather than spin vectors. Anisotropies appear
as meV per atom, meV per formula unit, joules per cubic metre, or as an effective field in tesla.
A number moved between two of those conventions without conversion is silently wrong and still
produces a plausible plot.

Every constant in this package therefore carries a declared unit string, and every public entry
point states the units of its arguments and its return value in the docstring. ``self_check()``
asserts the dimensional identities that the rest of the package relies on; it runs in CI.

The dimensional contract
------------------------
=====================  ==================================  ==========================
Symbol                 Meaning                             Unit
=====================  ==================================  ==========================
``mu``                 magnetic moment magnitude           J/T   (equivalently A m^2)
``gamma``              gyromagnetic ratio                  rad/(s T)
``alpha``              Gilbert damping                     dimensionless
``K``                  anisotropy energy scale             J
``xi``                 hard-axis to easy-axis ratio        dimensionless
``b``, ``B``           magnetic field                      T
``t``, ``T``           time, switching time                s
``tau0 = mu/(2 gamma K)``   Larmor timescale               s
``K/mu``               anisotropy field                    T
``Phi = int |b|^2 dt`` switching cost                      T^2 s
=====================  ==================================  ==========================

``Phi`` is **not** an energy. It is the time integral of the squared control field, which is
proportional to the Joule heating of the circuit that generates the field. Converting it to joules
requires a circuit constant with units J/(T^2 s) that encodes inductor geometry, resistance and
coupling efficiency. See :func:`cost_to_joules` and :class:`CircuitModel`, which force that
assumption to be stated rather than buried.

Constant values are CODATA 2022 recommended values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BOHR_MAGNETON_J_PER_T",
    "BOLTZMANN_J_PER_K",
    "ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T",
    "ELEMENTARY_CHARGE_C",
    "HBAR_J_S",
    "MEV_IN_JOULE",
    "UNITS",
    "CircuitModel",
    "bohr_magnetons_to_j_per_t",
    "cost_to_joules",
    "joules_to_mev",
    "landauer_limit_j",
    "mev_to_joules",
    "self_check",
    "tesla_to_rad_per_s",
    "thermal_energy_j",
]

# --------------------------------------------------------------------------------------
# Constants. Each one is named with its unit and is never written as a bare literal
# anywhere else in the package; scripts/check_unit_constants.py enforces that.
# --------------------------------------------------------------------------------------

#: Bohr magneton. CODATA 2022. Unit: J/T.
BOHR_MAGNETON_J_PER_T = 9.2740100657e-24

#: Boltzmann constant. Exact by SI definition since 2019. Unit: J/K.
BOLTZMANN_J_PER_K = 1.380649e-23

#: Electron gyromagnetic ratio. CODATA 2022. Unit: rad/(s T).
#: This is ``g_e mu_B / hbar`` to within the sign convention; the positive magnitude is used and
#: the sign of the precession is carried explicitly by the Landau-Lifshitz-Gilbert equation.
ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T = 1.76085962784e11

#: Elementary charge. Exact by SI definition since 2019. Unit: C.
ELEMENTARY_CHARGE_C = 1.602176634e-19

#: Reduced Planck constant. Exact derived value. Unit: J s.
HBAR_J_S = 1.054571817e-34

#: One millielectronvolt expressed in joules. Derived from the exact elementary charge.
MEV_IN_JOULE = ELEMENTARY_CHARGE_C * 1e-3

#: The declared unit of every quantity this package passes around. Consulted by the docs build and
#: by :func:`self_check`; a key added here without a corresponding docstring is a defect.
UNITS: dict[str, str] = {
    "mu": "J/T",
    "gamma": "rad/(s T)",
    "alpha": "1",
    "K": "J",
    "xi": "1",
    "b": "T",
    "t": "s",
    "T_switch": "s",
    "tau0": "s",
    "anisotropy_field": "T",
    "Phi": "T^2 s",
    "Phi_infinity": "T^2 s",
    "j": "A/m^2",
    "theta": "rad",
    "phi": "rad",
    "energy": "J",
    "temperature": "K",
}


# --------------------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------------------


def mev_to_joules(value_mev: float) -> float:
    """Convert an energy in millielectronvolts to joules.

    Args:
        value_mev: energy in meV.

    Returns:
        The same energy in J.
    """
    return value_mev * MEV_IN_JOULE


def joules_to_mev(value_j: float) -> float:
    """Convert an energy in joules to millielectronvolts.

    Args:
        value_j: energy in J.

    Returns:
        The same energy in meV.
    """
    return value_j / MEV_IN_JOULE


def bohr_magnetons_to_j_per_t(value_mu_b: float) -> float:
    """Convert a magnetic moment in Bohr magnetons to J/T.

    Args:
        value_mu_b: magnetic moment in units of the Bohr magneton.

    Returns:
        The same moment in J/T, which is the unit ``mu`` carries throughout this package.
    """
    return value_mu_b * BOHR_MAGNETON_J_PER_T


def tesla_to_rad_per_s(field_t: float, gamma_rad_per_s_t: float) -> float:
    """Convert a field to the Larmor angular frequency it drives.

    Args:
        field_t: magnetic field in T.
        gamma_rad_per_s_t: gyromagnetic ratio in rad/(s T).

    Returns:
        Angular frequency in rad/s.
    """
    return field_t * gamma_rad_per_s_t


def thermal_energy_j(temperature_k: float) -> float:
    """Thermal energy ``k_B T``.

    Args:
        temperature_k: temperature in K.

    Returns:
        Energy in J. This is the ``Theta`` that the thermal stability factor
        ``Delta = Delta_E / Theta`` divides by.
    """
    return BOLTZMANN_J_PER_K * temperature_k


def landauer_limit_j(temperature_k: float) -> float:
    """The Landauer limit ``k_B T ln 2``, the thermodynamic cost of erasing one bit.

    Args:
        temperature_k: temperature in K.

    Returns:
        Energy in J. At 300 K this is 2.87e-21 J, that is 2.87 zJ. It is a reference line, never a
        claim of proximity: a switching protocol is not an erasure.
    """
    return BOLTZMANN_J_PER_K * temperature_k * math.log(2.0)


# --------------------------------------------------------------------------------------
# The circuit model, which is where Phi becomes joules and where the assumption lives
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitModel:
    """The assumption that turns a switching cost into an energy.

    ``Phi = int |b(t)|^2 dt`` has units of T^2 s. The energy dissipated by the circuit that
    generates ``b`` is proportional to it, with a constant of proportionality set by the geometry
    and resistance of that circuit. This class exists so that constant is a named, printed,
    user-adjustable assumption instead of a buried literal.

    Attributes:
        joules_per_tesla2_second: the proportionality constant, unit J/(T^2 s).
        description: a human-readable statement of what circuit this represents. Required, because
            a number without it is not reportable.

    Example:
        A field coil of resistance ``R`` producing ``b = c I`` dissipates ``R I^2 = (R / c^2) b^2``,
        so ``joules_per_tesla2_second = R / c^2``.
    """

    joules_per_tesla2_second: float
    description: str

    def __post_init__(self) -> None:
        if self.joules_per_tesla2_second <= 0.0:
            raise ValueError("joules_per_tesla2_second must be positive")
        if not self.description.strip():
            raise ValueError(
                "a CircuitModel must carry a description; an unstated circuit assumption is not "
                "reportable, see docs/theory/03-what-the-cost-functional-measures.md"
            )


def cost_to_joules(phi_t2_s: float, circuit: CircuitModel) -> float:
    """Convert a switching cost to an energy through an explicit circuit model.

    Args:
        phi_t2_s: the switching cost ``Phi`` in T^2 s.
        circuit: the circuit assumption. There is no default, deliberately.

    Returns:
        Energy in J.

    Raises:
        TypeError: if ``circuit`` is not a :class:`CircuitModel`. Passing a bare float here is the
            exact mistake this signature exists to prevent.
    """
    if not isinstance(circuit, CircuitModel):
        raise TypeError(
            "cost_to_joules requires a CircuitModel, not a bare number. Phi is in T^2 s and only "
            "becomes joules through a stated circuit assumption."
        )
    return phi_t2_s * circuit.joules_per_tesla2_second


# --------------------------------------------------------------------------------------
# The self check. Asserts the dimensional identities the package relies on.
# --------------------------------------------------------------------------------------


def self_check() -> None:
    """Assert the dimensional identities this package depends on.

    Raises:
        AssertionError: if any identity fails. Run in CI; a failure here means every number the
            package produces downstream is suspect.
    """
    # A round trip through the energy conversions must be exact to floating point.
    for value_mev in (1.0, 3.38, 0.31, 1e-3):
        assert math.isclose(joules_to_mev(mev_to_joules(value_mev)), value_mev, rel_tol=1e-15)

    # tau0 = mu / (2 gamma K) must come out in seconds and be positive for physical inputs.
    mu = bohr_magnetons_to_j_per_t(2.0)  # J/T
    gamma = ELECTRON_GYROMAGNETIC_RATIO_RAD_PER_S_T  # rad/(s T)
    anisotropy_j = mev_to_joules(0.31)  # J
    tau0 = mu / (2.0 * gamma * anisotropy_j)
    assert tau0 > 0.0
    # For a 0.31 meV anisotropy on a 2 Bohr-magneton moment the Larmor timescale is sub-picosecond,
    # which is the regime the whole field operates in. A value outside [1e-15, 1e-9] s means the
    # inputs were handed over in the wrong units.
    assert 1e-15 < tau0 < 1e-9, f"tau0 = {tau0} s is outside the physical range for a 2D magnet"

    # The anisotropy field K/mu must come out in tesla, and for these inputs in the tens of tesla.
    anisotropy_field = anisotropy_j / mu
    assert 1e-3 < anisotropy_field < 1e4

    # Phi_infinity = 4 alpha K / (gamma mu) must come out in T^2 s.
    phi_infinity = 4.0 * 0.01 * anisotropy_j / (gamma * mu)
    assert phi_infinity > 0.0

    # The Landauer limit at 300 K is 2.87e-21 J. This is the one number in the package that is
    # widely quoted, so it is asserted against its published value rather than merely computed.
    assert math.isclose(landauer_limit_j(300.0), 2.8709e-21, rel_tol=1e-4)

    # A circuit model must refuse to exist without a description.
    try:
        CircuitModel(joules_per_tesla2_second=1.0, description="   ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("CircuitModel accepted an empty description")

    # cost_to_joules must refuse a bare float.
    try:
        cost_to_joules(1.0, 2.0)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("cost_to_joules accepted a bare number as the circuit model")
