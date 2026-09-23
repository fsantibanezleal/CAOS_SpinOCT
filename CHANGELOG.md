# Changelog

All notable changes to `spinoct` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), newest on top. Versions use the padded
display form `X.XX.XXX`; the PyPI/semver form drops the padding.

## [0.19.000] - 2026-09-22

### Added
- **A batched lane**: `spinoct.numeric.BatchImageOCPSolver` solves many independent optimal control
  problems as one tensor, on a GPU when there is one, behind the optional `spinoct[torch]` extra. One
  optimal control path is a small problem and a GPU does nothing for it; what the sweeps in front of
  this package do is solve hundreds of them that differ only in their parameters, and that is the shape
  the lane is for. The systems in a batch may differ in every parameter, and `solve_best` sweeps seeds
  as more rows of the same tensor.
- The batched gradient is exact (automatic differentiation) rather than the CPU lane's local finite
  difference; the two agree to 4.6e-06 of the gradient scale, which is that difference's own floor.
- Convergence is reported **per problem**, against a residual measured on the CPU lane's own paths
  rather than chosen: the reference's converged solves sit between 3.7e-06 and 5.5e-05, a stalled one
  at 2.2e-03, so the threshold is 1e-4. A batch is minimized jointly and a single verdict for all of it
  would hide the rows that did not make it.
- Accepted against both lanes that already have a gate: the closed-form uniaxial cost, and the CPU
  solver problem by problem on biaxial systems, where the worst deviation over a 72-problem sweep was
  8.5e-07. `docs/theory/14-the-batched-lane.md` has the method, the acceptance and the measured
  crossover; torch stays optional and CI does not install it.

### Fixed
- **`UniaxialOptimalControl` gained `peak_amplitude()`, because the obvious way to get the peak is
  wrong.** The pulse amplitude is `dn(u|m) + a p sn(u|m)` over `u` from 0 to `2K(m)`, and at both of
  those ends `sn = 0` and `dn = 1`: sampling the start and the midpoint of the pulse returns the same
  number, and for the negative parameter of a damped reversal that number is the pulse's MINIMUM. The
  peak is at a quarter of the way through, where `sn = 1`, and the closed form for it is
  `K / (mu p sqrt(1+a^2)) [sqrt(1 + a^2 p^2) + a p]`. Sampling the ends understates it by 0.2 per cent
  at alpha = 0.01 and T = 1 tau0, and by 37 per cent at alpha = 0.1 and T = 20 tau0. A driver sized on
  the old number would have been under-specified by a third. The new method is exact, not sampled, and
  a test holds it to a 200,001-point scan of the pulse.

## [0.18.000] - 2026-09-22

### Fixed
- **The longitudinal stabilizing field was applied with the wrong sign**, so the module's own analysis
  and its simulation contradicted each other. `perturbation_eigenvalues` calls a positive `B_r`
  stabilizing, and at one anisotropy field `hyperbolic_fraction` duly reported the instability gone,
  while `br_cost_reliability_front` added that field along the nominal moment and made switching LESS
  reliable. Measured on a 3 Bohr-magneton, 0.15 meV macrospin at a stability factor of one and
  alpha = 0.01: the success rate went from 0.780 without the field down to 0.530 with it, and now rises
  to 0.958. Two trajectories integrated side by side separate by a factor of 8.8 over the pulse under
  the old sign, 2.9 with no longitudinal field at all, and 1.2 under the corrected one.
- A test now holds the two together: the field the eigenvalues call stabilizing must raise the measured
  success rate, beyond its own confidence interval.

### Notes
- Every reliability front computed before this version understates what the longitudinal field buys.
  Espira's `novel.json` and case C09 are rebaked on 0.18.000, and manuscript M1, which reads that front,
  needs a revision.

## [0.17.000] - 2026-09-22

### Added
- `spinoct.lattice.recommended_images`: how many images a minimum energy path needs to resolve the wall
  as it travels, about three per wall width, odd, between 33 and 257. The reaction coordinate of a wall
  reversal is the wall's position, so a path whose images are further apart than the wall itself cannot
  follow it: the climbing image hops between lattice positions instead of settling on the saddle.
  Measured on a 32 x 32 patch at J / K = 2.5, whose wall is 1.12 sites wide and travels 32 sites: at the
  old fixed 33 images (0.9 per wall width) the force criterion was never met in 200,000 iterations
  (23 minutes) and the barrier wandered by about one part in a thousand; at 65 images it still was not;
  at the recommended 87 the same path converges in 1,185 iterations and 30 seconds, and agrees with a
  97-image path to seven digits (0.13460306 against 0.13460297 of N K).

### Changed
- CI runs the test suite on one Python version, the declared floor (ADR-0074 rule 3), with the wheel
  smoke still on 3.12.

### Notes
- `minimum_energy_path` keeps its default of 33 images, so no existing result moves silently; callers
  that sweep lattice sizes should pass `recommended_images(lattice)`.

## [0.16.000] - 2026-09-18

### Added
- `spinoct.lattice.SpinPatch`: a two-dimensional `W x H` square patch with nearest-neighbour exchange and
  uniaxial anisotropy. The free optimal-control solver and the string method run on it unchanged: both
  now reach the lattice through five operations (the batched field, energy and gradient; a colouring
  with disjoint closed neighbourhoods; the footprint sum; a wall coordinate; the coordination), which the
  chain implements with its original arithmetic. A chain solve and its barrier are bit-identical before
  and after (cost 1.0513818230273711e-10, barrier 2.090615508830735e-22 J, same iteration counts).
- The five-colour gradient on the patch, `(x + 2 y) mod 5`: 60 cost evaluations for the exact gradient
  at any patch size, checked against a brute-force gradient.
- A first two-dimensional result on theory page 12: on a 16 x 4 patch at long switching time the
  wall-mediated reversal reaches 0.657 of the uniform bound (still descending), so the chain's result
  carries over to a patch.

### Fixed
- The string method computed a chain's energy and forces whatever lattice it was given. Found by the
  first patch run, which returned the saturated chain barrier for every patch size; a straight wall now
  costs exactly its height times the chain barrier, as it must.

## [0.15.000] - 2026-09-18

### Fixed
- **The minimum energy path did not converge for wide domain walls.** The string method takes a forward
  step on the perpendicular force, and a fixed step of 0.02 (in units of the anisotropy energy per site)
  crossed the explicit stability limit `step * (2 + 4 J / K) < 2` just above J / K = 24. The path
  oscillated, hit the iteration cap and still returned a barrier: 1.5 times the continuum wall energy at
  J / K = 25 and 43 times it at J / K = 40, with `converged = False` as the only signal. Ten times the
  iterations did not help. The step now scales with the stiffness at 0.84 of the limit, which reproduces
  the old step at J / K = 10, where every barrier Espira and manuscript M2 use was computed (all nine
  recomputed: converged and identical).

### Added
- The continuum cross-check on theory page 12: the lattice barrier approaches the Bloch-wall energy
  `2 sqrt(2 J K)` from below as the wall widens, 0.955 of it at J / K = 2 to 0.999 at 80, with the
  deficit falling as `0.043 / w^2`, the leading discreteness correction. A test asserts convergence, the
  limit, and the `w^-2` rate.

## [0.14.000] - 2026-09-18

### Fixed
- **The spin-orbit-torque integrator solved a different equation from the one it cites.** The source
  (Vlasov et al., Phys. Rev. B 105, 134404, Eq. 3) writes the torques inside the implicit Gilbert form
  `s_dot = tau + alpha s x s_dot`; the integrator substituted the couplings directly into the explicit
  form, which drops the `alpha s x tau` mixing of the two channels. Measured against a direct linear
  solve of the implicit equation: 2 to 22 per cent off at alpha = 0.1, most where the damping-like
  coupling dominates. The explicit coefficients are now `xi_F - alpha xi_D` (field-like) and
  `xi_D + alpha xi_F` (damping-like), from `spinoct.dynamics.sot_torque.explicit_sot_coefficients`,
  shared by the integrator, the hybrid solver, its adjoint and the stochastic ensemble; the source's
  sweet spot falls out of it (at the ideal ratio the explicit damping-like term vanishes). A test
  compares the integrator with the implicit equation to machine precision.
- Both switching-success ensembles crashed at zero temperature (a division by the thermal energy). The
  stability factor there is reported as infinite.

### Added
- `spinoct.analytic.sot.ChirpedRotatingCurrent` (rung R04): the source's simplified constant-amplitude,
  linearly chirped rotating current (Eq. 15), with `resonant_frequency`, `characteristic_switching_time`
  and `at_source_settings`.
- `spinoct.thermal.sot_switching_success_rate` and an optional current in `stochastic_llg_step`: the
  thermal ensemble for current pulses, same thermostat and Heun scheme as the field ensemble.
- **A replication that does not reproduce.** The source reports switching probabilities of 0.89, 0.97
  and about 1 at 0.17, 0.18 and 0.20 j0 for the chirped current at a stability factor of 60. This engine
  gives 0.009, 0.043 and 0.22 at those amplitudes and 0.90 at 0.25 j0, the same curve shifted by about
  1.4 in amplitude. Rotation sense, starting tilt, coupling convention, chirp tuning, thermal noise,
  pulse length and a factor of two in the time unit were each checked and ruled out. The gap is
  recorded on theory page 09 and pinned in `tests/test_sot_chirp.py`.

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
