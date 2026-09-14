# The free chain optimal control path and the barrier floor

[08](08-beyond-the-macrospin.md) compared two fixed reversal modes of a spin chain. This page removes the
ansatz: every site follows its own trajectory, the switching cost is minimized over all of them, and
the answer is bracketed from below by an energy-barrier floor that holds for every pulse at every
switching time. The open problem is the one the method's authors state in print (Phys. Rev. B 107,
214448, 2023, https://doi.org/10.1103/PhysRevB.107.214448): under what conditions do nonuniform
switching mechanisms become optimal in terms of energy efficiency.

## The free optimal control path (`spinoct.lattice.LatticeOCPSolver`)

The image-based direct minimization of the macrospin solver ([05](05-numerical-optimal-control-path.md))
generalized to `N` coupled sites. The trajectory is `Q + 2` time slices of `N` unit vectors, both ends
clamped (all up, all down). On every interval and every site the midpoint rule gives a position and a
velocity, and the field that site needs comes from inverting the equation of motion with the chain's
internal field (anisotropy plus exchange from the neighbours):

```
b_i = (alpha / gamma) v_i + (1 / gamma) (s_i x v_i) - b_int,i_perp
Phi = sum_q sum_i |b_i|^2 dt
```

### Gradient by graph coloring

An interior image `(q, i)` enters two intervals in time and, through the exchange field, three sites in
space. Images two apart in time and three apart in space therefore have disjoint footprints, so one
perturbation of a whole color class gives every member's central difference at once: `2 x 3` classes,
times three components, times two signs, is 36 cost evaluations per gradient, independent of `Q` and
`N`. It matches a naive per-image central difference to a relative error of order 1e-8 (test gate).

### Optimizer

L-BFGS over unconstrained vectors `x` with `s = x / |x|`, the objective divided by the cost of the
analytic uniform path so the optimizer works in order-unity numbers. It converges about ten times
faster than the projected-descent reference, which is kept in the code.

### Three numerical traps, each found by a failing run and each now guarded

1. **The geodesic angle.** `arccos(l . r)` loses all precision when neighbouring images nearly coincide
   (a site parked at a pole), which turns the finite-difference gradient there into rounding noise.
   The solver uses `2 atan2(|r - l|, |r + l|)`, exact at every angle. The macrospin solver was changed
   the same way.
2. **The objective scale.** SciPy's relative-reduction test divides by `max(|f|, 1)`. Scaling the
   objective by the seed's own cost made the optimum look like `f << 1` for an expensive seed, and the
   test passed at 85 times the bound. The scale is now the analytic uniform-path cost.
3. **The wall seed.** Staggering the full optimal path in time compresses its precession into each
   site's short flip window, so a site jumps almost antipodally between consecutive images, where the
   midpoint rule is singular. That seed started at 12000 times the bound and pinned the optimizer at
   51. The wall seed now staggers only the polar angle and keeps the whole chain on the optimum's
   azimuth; a test asserts no consecutive-image step exceeds 0.5 rad.

### The upper bound

Uniform rotation is a feasible chain trajectory whose exchange field is purely longitudinal, so it
drops out of the transverse projection and the chain cost is exactly `N` times the single-site cost.
`uniform_images()` solves one site on the same grid, seeded from the closed-form optimum (Phys. Rev.
Lett. 126, 177206, 2021, https://doi.org/10.1103/PhysRevLett.126.177206), and repeats it. A solve that
starts there can only go down, so `cost <= uniform_bound` is exact, not a tolerance.

## The barrier floor (`spinoct.lattice.minimum_energy_path`)

Dot the inverted equation of motion with `s_dot`. The internal field does work at the rate of the site
energy change, so

```
mu b . s_dot = dE/dt + mu (alpha / gamma) |s_dot|^2.
```

Only the component of `b` along `s_dot` enters, so `|b|^2 >= (b . s_dot_hat)^2`, and on any stretch
where the energy rises the arithmetic-geometric mean inequality gives

```
|b|^2 >= (4 alpha / (gamma mu)) dE/dt.
```

Summed over sites and integrated over the rise, every reversal costs at least

```
Phi >= (4 alpha / (gamma mu)) Delta E_MEP
```

with `Delta E_MEP` the barrier of the minimum energy path between the two states, because every path
must climb at least that high. The bound holds at every switching time. For one uniaxial site
`Delta E = K` and the floor is exactly the known infinite-time optimum `Phi_inf = 4 alpha K / (gamma mu)`,
so it is tight there (test gate, to 1e-6).

Equality needs two things at once: the field parallel to the velocity, and the precession supplied for
free by the internal field. Uniform rotation of a uniaxial chain satisfies both in the long-time limit.
A domain wall does not obviously satisfy the second: sites inside the wall see different internal
fields and would precess at different rates. So the floor is rigorous, but tightness for a wall is not
guaranteed.

The MEP is found by the climbing-image string method (W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys.
126, 164103, 2007, https://doi.org/10.1063/1.2720838) on the product of unit spheres, in the geodesic
form used for magnetic systems (P. F. Bessarab, V. M. Uzdin, H. Jonsson, Comput. Phys. Commun. 196,
335, 2015, https://doi.org/10.1016/j.cpc.2015.07.001).

### Barriers, J/K = 10

| N | barrier / K | barrier / (N K) |
|---|---|---|
| 4 | 4.0000 | 1.0000 |
| 8 | 7.7475 | 0.9684 |
| 12 | 8.6991 | 0.7249 |
| 16 | 8.8396 | 0.5525 |
| 24 | 8.8665 | 0.3694 |
| 32 | 8.8672 | 0.2771 |

Short chains reverse coherently (barrier `N K`). Above about eight sites a wall entering at an open end
is cheaper in energy, and the barrier saturates at 8.867 K, against the continuum wall energy
`2 sqrt(2 J K) = 8.944 K` for the energy density `(J/2)(d theta/dx)^2 + K sin^2 theta`. The discrete
chain sits slightly below the continuum value, as it should.

## The result: above a length and a time, the optimal reversal is a domain wall

Uniform rotation is not always the cheapest reversal. For a chain longer than the barrier crossover
length, at a long enough switching time, the free optimal control path nucleates a domain wall at an
open end and sweeps it through, and it is strictly cheaper than rotating the chain uniformly.

This reverses the conclusion of [08](08-beyond-the-macrospin.md), which compared uniform rotation only
with a constant-speed wall at a fixed short time. That comparison was correct for what it tested; the
free search shows the ansatz, not the physics, was the limit.

### The evidence (J/K = 10, alpha = 0.1)

Ratio of the cheapest trajectory found to the uniform bound on the same grid. Every value is the cost
of an explicit feasible trajectory, so it is an upper bound on the true optimum over uniform rotation
whether or not the optimizer converged.

| T / tau0 | N = 4 | N = 8 | N = 12 | N = 16 | N = 24 | N = 32 |
|---|---|---|---|---|---|---|
| 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 150 | 1.0000 | 1.0000 | **0.864** | **0.899** (0.838 at 600 images) | 1.0000 | 1.0000 |
| barrier / (N K) | 1.000 | 0.968 | 0.725 | 0.552 | 0.369 | 0.277 |

Images per grid: 49 to 74 at T = 20 tau0, 120 at 60 tau0, 198 at 150 tau0. At T = 150 tau0 the uniform
cost is within 0.2 percent of its infinite-time value, so the floor ratio there is the barrier ratio
to that accuracy.

At N = 12 and 16 and T = 150 tau0 the optimum is nonuniform (nonuniformity about 0.5 rad across the
chain) and saves at least 14 and 16 percent. At N = 24 and 32 no start found a trajectory under the
bound within the iteration budget at T = 150 tau0: the wall must cross more sites, and the time a
wall needs grows with the chain. Those entries are "not found", not "uniform is optimal"; the floor
leaves room for savings of up to 73 percent there.

At N = 16 and T = 150 tau0 the uniform optimum itself stops being a local minimum: a start perturbed
from it by 0.05 rad drifts below the bound and away from uniformity, where at every shorter time and
every shorter chain the same perturbation relaxes back. That is the symmetry-breaking instability that
opens the nonuniform branch.

### How the N = 16 result was verified

The first sub-uniform trajectory came from the MEP seed at 300 images (ratio 0.892). It was checked
three independent ways before being believed:

1. **Grid refinement.** Interpolated to 600 images and re-optimized, the ratio fell to 0.838 against
   the 600-image bound. A discretization artifact would shrink under refinement; this saving grew.
2. **Local consistency.** Each interval was integrated from its left image under its own midpoint
   field with 50 RK4 substeps and compared with its right image. The wall trajectory's local error
   (max 4.1e-3, mean 7.1e-4) is the same order as the uniform optimum's (max 2.0e-3, mean 6.5e-4) on
   the same grid, so the discrete cost is as faithful for the wall as for uniform rotation.
3. **Open-loop dynamics.** The recovered per-site fields of the 600-image trajectory were applied to
   the chain equation of motion from the all-up state with a fine RK4 integrator. Every site reverses
   (final s_z = -1.000 on all 16) and the simulated motion tracks the optimized trajectory to within
   0.106. The pulse is realizable, not only a discrete cost.

A second, unrelated start confirms it: the tanh-wall seed alone reaches 0.864 at N = 12 and 0.899 at
N = 16 on a 198-image grid.

### Where the long-time regime starts

The uniform optimum reaches its infinite-time floor on a time scale set by the damping, so stronger
damping reaches the wall regime at shorter switching time. At alpha = 0.5, J/K = 6, a ten-site chain
at T = 40 tau0 already saves 19 percent (ratio 0.810 in 400 iterations); the package test suite
asserts a saving above 10 percent in that configuration and none for a four-site chain. The
crossover map over (N, T) at two damping values is baked by the Espira product
(https://github.com/fsantibanezleal/CAOS_RES_Espira, `data-pipeline/run_lattice_ocp.py`).

### What is and is not established

Established: a nonuniform reversal strictly cheaper than uniform rotation exists for these chains,
verified by refinement, local consistency and forward dynamics; the barrier floor is rigorous at every
switching time and tight for the single site. Not established: that the trajectories found are the
global optima (the optimizer reports "not converged" for most nonuniform runs, so the true savings are
at least the values quoted), and whether the floor is tight for walls in the long-time limit.

## Sources

- M. H. A. Badarneh, G. J. Kwiatkowski, P. F. Bessarab, Phys. Rev. B 107, 214448 (2023),
  https://doi.org/10.1103/PhysRevB.107.214448. The image-based method and the open beyond-macrospin
  question.
- G. J. Kwiatkowski, M. H. A. Badarneh, D. V. Berkov, P. F. Bessarab, Phys. Rev. Lett. 126, 177206
  (2021), https://doi.org/10.1103/PhysRevLett.126.177206. The closed-form uniaxial optimum and
  `Phi_inf`.
- W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838.
  The simplified string method.
- P. F. Bessarab, V. M. Uzdin, H. Jonsson, Comput. Phys. Commun. 196, 335 (2015),
  https://doi.org/10.1016/j.cpc.2015.07.001. Geodesic minimum energy paths for magnetic systems.
