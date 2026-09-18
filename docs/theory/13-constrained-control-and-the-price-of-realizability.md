# Constrained optimal control and the price of realizability

The analytic and image-based optimal control paths minimize the cost with no constraint on the control,
so their pulses are mathematically optimal but not necessarily producible: the amplitude and the phase
vary on the Larmor timescale. A real driver has a finite peak amplitude and a finite bandwidth. The
question an experimentalist actually asks is what the cost becomes once the pulse must be realizable,
and answering it requires optimizing the control **directly**, because only a direct parameterization
can carry a constraint at all.

Two parameterizations, both discretize-then-optimize:

- **GRAPE** (gradient ascent pulse engineering): the transverse field is piecewise linear between
  $N$ slice values, and the slice values are optimized under a box constraint on the amplitude and an
  optional penalty on the slew rate. Standard in magnetic resonance, essentially unused for classical
  magnetization dynamics.
- **CRAB** (chopped random basis): the field is a truncated sum of a few Fourier components whose base
  frequencies are the harmonics of the switching window, nudged randomly. The truncation *is* the
  bandwidth limit, so the result is realizable by construction rather than by penalty.

Both integrate the same Landau-Lifshitz-Gilbert equation as the rest of the package and report the same
switching cost, so their pulses sit on the same axes as the unconstrained optima.

## The objective, and why it needs a continuation

The objective is the cost plus a penalty for an incomplete reversal,

$$L = \Phi + \lambda \, \frac{1 + s_z(T)}{2},$$

where the infidelity $(1 + s_z(T))/2$ is zero for a complete reversal to the south pole and one for no
reversal at all.

A single solve at a fixed $\lambda$ does not answer the question. The two terms trade against each
other, so the cheapest point of $L$ is a pulse that leaves the moment part way: at one harmonic this
stopped at an infidelity of $0.15$, which is $s_z(T) = -0.69$, and reported its cost against an analytic
optimum that reverses exactly. Two different quantities were being compared.

The solvers therefore run a penalty continuation: solve to convergence, measure the infidelity, and if
it is above $10^{-5}$ multiply $\lambda$ by ten and restart from the current point. A pulse that never
reaches the target is reported as not having reversed, with its infidelity, rather than as a cheap cost.
The reporting threshold is $10^{-3}$, that is $s_z(T) \le -0.998$; the sign of $s_z$ alone is not
enough, because a pulse that stops at $s_z = -0.3$ is in the reversed basin and has not switched.

## The exact gradient through a linear basis

Both controls are **linear** in their parameters. On the integration grid the field is

$$b(t_i) = \sum_p D_{ip} \, \theta_p$$

for a fixed design matrix $D$: for GRAPE the interpolation weights from the slice nodes, for CRAB the
harmonic basis. The gradient with respect to the parameters is then one transposed multiply away from
the gradient with respect to the field,

$$\frac{\partial L}{\partial \theta} = D^{\mathsf T} \frac{\partial L}{\partial b},$$

and $\partial L / \partial b$ is exactly what the discrete adjoint of page 10 returns in a single
backward pass, at the cost of one extra integration and independent of the number of parameters. The
optimizer runs against the adjoint's own forward integration, so the objective and the gradient are a
consistent pair; the final pulse is re-evaluated with the package's norm-preserving RK4 for reporting.
At the grid used (2400 steps) the two integrators differ by about $4\times10^{-5}$ in the final $s_z$,
far inside the reversal threshold, and the reported cost $\Phi = \int |b|^2 \, dt$ does not depend on
the integrator at all.

The same construction covers the hybrid field-plus-current problem of page 9, with the adjoint extended
to the spin-orbit-torque terms: every term of that right-hand side is a cross product, so every Jacobian
is a product of skew matrices, and one backward pass returns the gradient with respect to both controls
at once.

## What this replaced, and the test that let it through

The first versions had no gradient. GRAPE ran L-BFGS-B on a finite-difference gradient, which costs one
forward integration per parameter and is noisy at the integration tolerance; CRAB and the hybrid solver
ran a Nelder-Mead simplex, which does not converge over a few dozen parameters. Measured on the uniaxial
oracle at ten $\tau_0$, CRAB took 90 to 230 seconds per solve and returned **2.2 times** the analytic
optimum at two harmonics rising to **14 times** at six. More harmonics is a strictly larger feasible
set, so the best achievable cost cannot rise with bandwidth: that monotonicity is the falsifier, and it
is what identified the optimizer rather than the physics as the problem.

The test suite had a monotonicity test and it passed throughout. It asserted that a richer basis
reversed "at least as completely" as a poorer one, with a slack of $0.2$ in the final $s_z$, which
almost anything satisfies. It was testing the wrong quantity with a tolerance wide enough to hide the
defect. The test now asserts the inequality on the **cost**, which is what the method reports, and
requires every answer to be a real reversal first.

A second defect surfaced once the gradient worked. The CRAB basis carried one sine per harmonic, so the
$x$ and $y$ components shared a phase at every frequency: the drive was linear in the plane and could
not rotate. The optimal uniaxial pulse **is** a rotating field, so the basis paid a large and entirely
artificial penalty, sitting at 6.2 times the analytic optimum and not improving with bandwidth, because
no number of harmonics turns a linear drive into a rotating one. The basis now carries both quadratures.

## What it measures now

On the reference macrospin (3 Bohr magnetons, 0.15 meV, $\alpha = 0.1$) at ten $\tau_0$, against the
closed-form optimum, measured 2026-09-17:

| Harmonics | CRAB cost / analytic optimum |
|---|---|
| 1 | 2.15 |
| 2 | 1.39 |
| 3 | 1.21 |
| 4 | 1.16 |
| 6 | 1.13 |
| 8 | 1.14 |

Monotone down to a floor about 13 per cent above the unconstrained optimum, which is the genuine price
of a band-limited pulse that must vanish at both ends of the window. Each solve takes 10 to 90 seconds
where the simplex took 90 to 230 and did not converge.

Under an amplitude cap, GRAPE recovers the unconstrained optimum when the cap is loose (cost / optimum
$= 1.003$ at 20 anisotropy fields, a positive control that the direct solver and the closed form
describe the same problem), pays 1.08 as the cap approaches the peak amplitude the optimum itself uses
(0.63 anisotropy fields), and **cannot reverse the moment at all** below about 0.45 anisotropy fields
per component. That threshold is the honest answer at a tight cap, and it is reported as a pulse that
did not switch rather than as a cheap cost.

For the hybrid problem, sweeping the relative price of current from $10^{-1}$ down to $10^{-4}$ moves
the share of the weighted cost carried by the field from 0.96 to 0.03: the crossover from a
field-dominated to a current-dominated optimum is inside that window, and outside it the sweep is flat.

## Sources

GRAPE: Khaneja, Reiss, Kehlet, Schulte-Herbrueggen and Glaser, *Optimal control of coupled spin
dynamics: design of NMR pulse sequences by gradient ascent algorithms*, J. Magn. Reson. 172, 296 (2005),
https://doi.org/10.1016/j.jmr.2004.11.004.

CRAB: Caneva, Calarco and Montangero, *Chopped random-basis quantum optimizations*, Phys. Rev. A 84,
022326 (2011), https://doi.org/10.1103/PhysRevA.84.022326.

The adjoint construction and its sources are on page 10; the spin-orbit-torque dynamics and the hybrid
cost are on page 9.
