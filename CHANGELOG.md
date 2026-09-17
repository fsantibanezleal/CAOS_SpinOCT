# Changelog

All notable changes to `spinoct` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), newest on top. Versions use the padded
display form `X.XX.XXX`; the PyPI/semver form drops the padding.

## [0.13.000] - 2026-09-17

### Changed
- **The constrained solvers are driven by the exact adjoint gradient.** GRAPE, CRAB and the
  field-plus-current hybrid all parameterize the control linearly, so the gradient with respect to the
  parameters is one transposed multiply from the gradient with respect to the field, which the discrete
  adjoint returns in a single backward pass. GRAPE had been running L-BFGS-B on a finite-difference
  gradient; CRAB and the hybrid solver had been running a Nelder-Mead simplex over a few dozen
  parameters, which does not converge. Measured on the uniaxial oracle at ten tau0: CRAB took 90 to 230
  seconds per solve and returned 2.2 times the analytic optimum at two harmonics rising to 14 times at
  six, where a strictly larger search space cannot cost more. It now falls monotonically, 2.15 at one
  harmonic to 1.13 at six, in 10 to 90 seconds.
- **The CRAB basis carries both quadratures.** With one sine per harmonic the two transverse components
  shared a phase at every frequency, so the drive was linear in the plane and could not rotate. The
  optimal uniaxial pulse is a rotating field, so the basis paid an artificial penalty that no amount of
  bandwidth could remove: 6.2 times the analytic optimum, flat in the harmonic count.
- **A reported cost now belongs to a pulse that actually reversed the moment.** The penalized objective
  trades fidelity against cost, so a single solve at a fixed weight returns a cheap pulse that stops part
  way: at one harmonic it stopped at an infidelity of 0.15, that is s_z(T) = -0.69, and quoted its cost
  against an optimum that reverses exactly. The solvers now raise the penalty and restart until the
  infidelity is below 1e-5, and `switched` means `(1 + s_z(T)) / 2 <= 1e-3` rather than merely
  `s_z(T) < 0`.
- The constrained solvers integrate on 2400 steps by default, where the adjoint's forward Euler pass and
  the reporting RK4 pass agree to 4e-5 in the final s_z.

### Added
- `spinoct.adjoint.adjoint_gradient_sot`: the discrete adjoint extended to the spin-orbit-torque
  dynamics, returning the exact gradient with respect to the field AND the current in one backward pass.
  Verified against finite differences to one part in ten to the fifth.
- `spinoct.control.linear_basis`: the shared machinery for a control that is linear in its parameters
  (the design matrix, the chain rule, the penalty continuation), with the interpolation and harmonic
  bases.
- `docs/theory/13-constrained-control-and-the-price-of-realizability.md`: the missing theory page for
  the constrained rungs, with what they measure and the two defects that shipped before them.

### Fixed
- The monotonicity test that should have caught all of this. It asserted that a richer basis reversed
  "at least as completely" as a poorer one, with a slack of 0.2 in the final s_z, which almost anything
  satisfies. It now asserts the inequality on the cost, which is what the method reports, and requires
  every answer to be a real reversal first. A second test asserts that no constrained pulse can cost
  less than the analytic optimum.

## [0.12.000] - 2026-09-17

### Changed
- ImageOCPSolver.solve minimizes with L-BFGS over normalized vectors by default; the projected gradient
  descent with geodesic retraction stays available as `method="descent"` and as the reference. Measured
  against the closed form: at T = 10 tau0 the new default converges in 205 iterations where descent was
  still moving at 2500, and at T = 100 tau0 it reaches 1.5 per cent above the closed form in the budget
  where descent reached 12 per cent (and 4 per cent after sixteen times the iterations).
- The resolution rule tightened to 0.1 rad per interval, measured: 0.15 rad leaves 1.4 per cent of
  discretization error at T = 100 tau0 and 0.1 rad leaves 0.7 per cent, below which the iteration budget
  binds rather than the grid. Together the two changes keep the numerical optimum within about one per
  cent of the closed form across the switching times the product bakes, where it was 20 per cent adrift.

## [0.11.000] - 2026-09-17

### Added
- ImageOCPSolver.recommended_images: the image count that resolves the optimal path at a given switching
  time. The midpoint rule's error is set by how far the moment moves between images, and the optimal path
  spirals, so a fixed count degrades as the switching time grows. Measured on the uniaxial oracle at 60
  images: the numerical cost matches the closed form at T = 2 tau0 and sits 20 per cent above it at
  T = 100 tau0. The rule keeps the geodesic step under 0.15 rad per interval, with a floor of 60 images,
  and a test asserts both the regression and that the rule removes it.

## [0.10.000] - 2026-09-13

### Added
- spinoct.lattice.LatticeOCPSolver: the free optimal control path of a spin chain, the image-based
  direct minimization generalized to N coupled sites. Exact central-difference gradient by graph
  coloring (36 cost evaluations per gradient, independent of chain length and image count), L-BFGS on
  normalized vectors, analytic-optimum seeds (uniform, tanh wall on the optimum's azimuth, minimum
  energy path), and an exact upper bound (the single-site optimum on every site).
  recommended_images resolves both the precession and a wall crossing.
- spinoct.lattice.minimum_energy_path and cost_floor_from_barrier: the climbing-image geodesic string
  method, and the rigorous floor Phi >= 4 alpha dE_MEP / (gamma mu) that holds for every pulse at every
  switching time (tight for the single uniaxial site, where it equals Phi_inf).
- Finding: above the barrier crossover length and at long switching time the optimal reversal of a
  chain is a domain wall, strictly cheaper than uniform rotation (at least 16 percent for 16 sites,
  J/K = 10, alpha = 0.1, T = 150 tau0), verified by grid refinement, local consistency and open-loop
  forward dynamics. Supersedes the two-mode conclusion of docs/theory/08. docs/theory/12.

### Fixed
- The geodesic angle between neighbouring images is computed as 2 atan2(|r - l|, |r + l|) in both the
  chain and the macrospin image solvers. arccos of the dot product lost all precision for nearly
  coincident images and turned the finite-difference gradient there into rounding noise.

## [0.09.000] - 2026-09-13

### Added
- spinoct.amortized: an amortized learned policy (R15), a small numpy MLP that emits a near-optimal
  pulse for new material parameters instantly. Finding: the output representation decides whether
  amortization works. Regressing the raw amplitude profile fails (switching is threshold-sensitive,
  the emitted pulse is a few percent too weak and does not reverse); amortizing the physically
  meaningful shape parameter succeeds, emitting pulses that reverse the moment at within ten percent
  of the analytic optimum on held-out parameters. Passes the pre-declared acceptance gate.
  docs/theory/11.

## [0.08.000] - 2026-09-13

### Added
- spinoct.adjoint: exact gradient-based pulse optimization by the discrete adjoint method (R10). A
  hand-derived reverse-mode gradient through a norm-projected LLG integration, validated against
  finite differences to 1e-8, fed to L-BFGS-B which converges onto the analytic optimum (within a few
  percent of the closed-form cost). Pure numpy, no autodiff dependency; the reverse pass batches to a
  GPU tensor library unchanged. Non-dimensionalized so the optimizer works in order-unity variables.
  docs/theory/10.

## [0.07.000] - 2026-09-13

### Added
- spinoct.control.hybrid: joint field-plus-spin-orbit-torque optimal control (R13), the design space
  the kickoff paper names as open. A SOT-augmented norm-preserving integrator (integrate_llg_sot) and
  a HybridSolver that co-optimizes a band-limited field and current under a two-term cost
  C_b int|b|^2 + C_j int|j|^2. Validated: a pure damping-like current drives the moment to the equator
  (the SOT sign check), and the balanced co-optimization reverses using both controls. docs/theory/09.

## [0.06.000] - 2026-09-13

### Added
- spinoct.pareto: the multi-objective Pareto front (R14) over switching time, cost, peak field and
  spectral bandwidth, with dominance marking. The analytic optimal family trades switching time
  against the other three objectives, so it forms one continuous front; faster switching costs more
  peak field and bandwidth.

## [0.05.000] - 2026-09-13

### Added
- spinoct.lattice: optimal control beyond the macrospin (Gap 1). A ferromagnetic spin chain with
  nearest-neighbour exchange (SpinChain), validated against the analytic macrospin cost in the
  one-site limit, and a reversal-mode comparison (uniform rotation vs a domain-wall sweep). Finding:
  for the switching-cost metric, uniform rotation is the field-cost optimum across N=2-128 and
  J/K=0.2-10, because the domain wall forces fast local flips and pays exchange; domain walls
  dominate real switching for thermal-barrier reasons, not field-cost reasons. docs/theory/08.

## [0.04.000] - 2026-09-13

### Added
- spinoct.thermal: stochastic Landau-Lifshitz-Gilbert dynamics with the fluctuation-dissipation
  thermal field (validated against the Boltzmann distribution), the switching success rate over an
  ensemble (R11), and the longitudinal-field cost-reliability front plus the instability-penalized
  optimal control path (R12). The front reproduces the published half-hyperbolic bare path, the
  success dip near B_r = 0.5 K/mu, and unity at large field, and adds the cost of that reliability,
  a number not previously computed. docs/theory/07 authored.

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
