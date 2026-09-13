# Changelog

All notable changes to `spinoct` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), newest on top. Versions use the padded
display form `X.XX.XXX`; the PyPI/semver form drops the padding.

## [0.03.000] - 2026-09-13

### Added
- control.baselines: static antiparallel field, Sun-Wang minimal constant field, precessional pulse
  (R00-R04). analytic.sot: the closed-form optimal SOT protocol (R06). metrics.ProtocolMetrics: the
  dimensionless scoring suite (cost over floor and over the free cost, peak amplitude, spectral
  bandwidth).
- control.constrained: GRAPE (piecewise-constant field under an amplitude cap and slew penalty) and
  CRAB (band-limited randomized Fourier basis) (R08, R09). These optimize the control directly, which
  is what lets a realizability constraint be imposed, and answer the price-of-realizability question.
  A fast norm-preserving tabulated RK4 integrator backs the optimization loop.

## [0.02.000] - 2026-09-13

### Added
- `numeric.ImageOCPSolver`: the numerical image-based optimal control path solver (Badarneh,
  Kwiatkowski, Bessarab, Phys. Rev. B 107, 214448 (2023)). Represents the trajectory as a chain of
  unit vectors on the sphere, minimizes the midpoint-rule switching cost by projected gradient descent
  with geodesic retraction and a backtracking line search, and converges on a dimensionless
  relative-cost criterion. The gradient is local, so each iteration is O(Q) rather than O(Q^2).
- `ImageOCPSolver.solve_best`: a multi-seed sweep that keeps the lowest-cost path, the guard against
  reporting a symmetric local minimum when several optimal control paths coexist (the biaxial case).
- Symmetry-breaking of the initial chain is on by default, because the pole-to-pole geodesic is a
  saddle where the gradient vanishes and a solve started on it reports the meridian as converged.

### Validated
- The numerical cost reproduces the exact analytic uniaxial cost from above, with the gap falling as
  the image count rises (the acceptance gate). The numerical path precesses, matching the analytic
  peak-to-peak azimuth. A hard axis pushes the numerical cost below the free-macrospin floor, the
  central biaxial claim, which has no closed form.

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
