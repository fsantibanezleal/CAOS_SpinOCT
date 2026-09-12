# Positive controls

A numerical solver that agrees with itself proves nothing. The lesson, recorded across the CAOS
conventions, is that a fitted or iterated result can be confidently wrong in a way no self-consistency
check catches. `spinoct` guards against that by validating every numerical method against results that
are known in closed form, before the numerical method is allowed to produce a published number.

The uniaxial macrospin is where the closed form exists (Phys. Rev. Lett. 126, 177206 (2021),
https://doi.org/10.1103/PhysRevLett.126.177206). These are the identities `spinoct.analytic.uniaxial`
implements and `tests/test_uniaxial_analytic.py` asserts, each by an independent route.

## The exact identities

| Quantity | Closed form | How the test checks it |
|---|---|---|
| Free-macrospin cost | `Phi_f = pi^2 (1 + alpha^2) / (gamma^2 T)` | the zero-damping cost equals it exactly |
| Universal floor | `Phi_inf = 4 alpha K / (gamma mu)` | the long-time cost approaches it from above |
| Uniaxial cost | `Phi_m = 2K [2E(m) - K(m)] / (gamma mu p)` | equals the quadrature of the square of the pulse |
| Mean amplitude | `b_av = pi sqrt(1+a^2) / (gamma T)` | equals the trapezoidal mean of the sampled pulse |
| Amplitude spread | `Delta b = 2 alpha K / (mu sqrt(1+a^2))` | equals the sampled peak-to-trough spread |
| Peak times | `t_max = T/4`, `t_min = 3T/4` | equal the argmax and argmin of the sampled pulse |
| Pulse symmetry | `b(0) = b(T/2) = b(T)` | checked at the three times |

## The inequalities

- `Phi_m >= Phi_f` for a uniaxial magnet, with equality only at zero damping. Easy-axis anisotropy can
  only obstruct the reversal; it never helps. (A biaxial magnet is different, which is the whole point
  of the hard-axis mechanism, but that is a separate, numerical case.)
- `Phi_m >= Phi_inf` at any switching time.
- `Phi_m(T)` decreases monotonically in `T`.

## The physics round trip

The decisive check is not an identity between closed forms, it is a confrontation with the equation of
motion. The test `test_forward_integration_under_the_optimal_pulse_actually_reverses_the_moment` feeds
the analytic pulse to a norm-preserving integrator of the Landau-Lifshitz-Gilbert equation and confirms
the moment ends at the reversed state, along the predicted trajectory. And
`test_inverting_the_equation_of_motion_recovers_the_optimal_field` runs the inversion the derivation
depends on and recovers the pulse from the trajectory, with a residual that shrinks at the second-order
rate as the sampling grid is refined.

## Why the elliptic functions get their own controls

The pulse and cost are built on Jacobi elliptic functions at a **negative** parameter, `-alpha^2 p^2`,
which `scipy` does not evaluate directly. The reduction in `spinoct.analytic.elliptic` is a closed-form
transformation, so testing it against elliptic identities would be circular. Instead
`tests/test_elliptic.py` checks every value against a direct numerical inversion of the defining
incomplete integral, which shares no code with the transformation.
