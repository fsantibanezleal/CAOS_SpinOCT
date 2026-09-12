# Changelog

All notable changes to `spinoct` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), newest on top. Versions use the padded
display form `X.XX.XXX`; the PyPI/semver form drops the padding.

## [0.01.000] - 2026-09-12

### Added
- The dimensional contract in `spinoct.units`: CODATA 2022 constants, meV and Bohr-magneton
  conversions, the thermal and Landauer energies, and `CircuitModel`, which forces the assumption
  that turns a switching cost (tesla-squared-seconds) into an energy (joules) to be a named,
  described, positive object rather than a buried literal. `self_check()` asserts the dimensional
  identities and runs in CI.
- Negative-parameter Jacobi elliptic functions (`sn`, `cn`, `dn`, unwrapped `am`, complete `K` and
  `E`) via the imaginary-modulus transformation, validated against direct quadrature of the defining
  integral, which is the regime the optimal control path requires (parameter `-alpha^2 p^2 < 0`).
- `MacrospinSystem`: biaxial anisotropy energy, internal field and its transverse part, the Hessian,
  and the derived scales `tau0`, the anisotropy field, the energy barrier and the thermal stability
  factor.
- The Landau-Lifshitz-Gilbert equation, its inversion (the field that produces a given trajectory),
  the switching-cost quadrature, and a norm-preserving RK4 integrator.
- The exact uniaxial optimal control path (`UniaxialOptimalControl`): the trajectory, the closed-form
  perpendicular pulse and its Cartesian vector, and the closed-form cost, mean amplitude, amplitude
  spread and peak times. Reproduces the four published identities of Phys. Rev. Lett. 126, 177206
  (2021) and the fast, slow and universal-floor cost asymptotics, each asserted as a positive control.
- `SwitchingTimeTooLongError`: raised when the switching time is so long that the shape parameter is
  not representable in double precision, which is a physical statement (the cost has saturated onto
  its floor) rather than a numerical failure.
- CI guards: `check_unit_constants.py` (no unreviewed dimensional constant in solver code) and the
  ruff + pytest lanes.

[0.01.000]: https://github.com/fsantibanezleal/CAOS_SpinOCT/releases/tag/v0.01.000
